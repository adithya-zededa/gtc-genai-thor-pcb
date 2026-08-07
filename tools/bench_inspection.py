#!/usr/bin/env python3
"""Per-stage latency and output-validity benchmark for one PCB inspection.

Establishes the baseline that architectural changes (notably the planned
vision/policy model split) get measured against. Answers three questions the
architecture docs previously guessed at:

  1. How much of an inspection is prefill vs decode?
  2. Does structured decoding cost or save time, and what does it fix?
  3. How large is the image prefill that the tool loop used to repeat?

Usage::

    python tools/bench_inspection.py --url http://10.43.254.130:8000 \
        --image path/to/frame.jpg [--runs 5] [--json results.json]

If --image is omitted, a frame is pulled from the video simulator clip named
by CAMERA_VIDEO_SOURCE (or ./IMG_1043.MOV), choosing the frame with the
highest in-zone edge density so a board is actually present.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.vlm.prompts import DEFAULT_MONITORING_DEFECT_PROMPT  # noqa: E402
from agents.vlm.schemas import (  # noqa: E402
    PCB_INSPECTION_RESPONSE_SCHEMA,
    STRUCTURED_MAX_TOKENS,
)

REQUIRED_TOP = ("detected", "confidence", "reasoning", "should_alert")
REQUIRED_DETAIL = (
    "power_jack_status",
    "usb_port_status",
    "header_pins_status",
    "defects",
)


# ── frame acquisition ──────────────────────────────────────────────────────

def frame_from_video(path: Path, samples: int = 60) -> np.ndarray:
    """Pick the sampled frame with the most in-zone edge detail (i.e. a board)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    best_score, best_frame = -1.0, None
    for i in range(0, total, max(1, total // samples)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        roi = gray[int(h * 0.2):int(h * 0.85), :]
        score = float(np.mean(cv2.Canny(cv2.resize(roi, (160, 120)), 50, 150) > 0))
        if score > best_score:
            best_score, best_frame = score, frame
    cap.release()
    if best_frame is None:
        raise SystemExit(f"No readable frames in {path}")
    return best_frame


def encode(frame: np.ndarray, max_dim: int = 1024) -> Tuple[str, float]:
    """Resize + JPEG encode exactly as UnifiedVLMClient does. Returns (b64, seconds)."""
    start = time.perf_counter()
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise SystemExit("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode(), time.perf_counter() - start


# ── request helpers ────────────────────────────────────────────────────────

def build_payload(
    model: str, image_b64: str, *, max_tokens: int, schema: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": DEFAULT_MONITORING_DEFECT_PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "repetition_penalty": 1.15,
        "frequency_penalty": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "pcb_inspection", "schema": schema},
        }
    return payload


def measure(
    url: str, payload: Dict[str, Any], runs: int, warm: bool = True
) -> Dict[str, Any]:
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    if warm:
        requests.post(endpoint, json=payload, timeout=300).raise_for_status()

    latencies: List[float] = []
    completions: List[int] = []
    valid_json = typed_ok = detail_ok = 0
    sample = ""
    prompt_tokens = 0

    for i in range(runs):
        start = time.perf_counter()
        resp = requests.post(endpoint, json=payload, timeout=300)
        latencies.append(time.perf_counter() - start)
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        usage = body["usage"]
        completions.append(usage["completion_tokens"])
        prompt_tokens = usage["prompt_tokens"]
        if i == 0:
            sample = content

        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            continue
        valid_json += 1
        # "Typed" means the fields are the types downstream code assumes.
        # bool("missing") is True, so a string here silently becomes a defect
        # verdict — that is the failure this benchmark exists to catch.
        if (
            isinstance(obj.get("detected"), bool)
            and isinstance(obj.get("should_alert"), bool)
            and isinstance(obj.get("confidence"), (int, float))
            and isinstance(obj.get("reasoning"), str)
        ):
            typed_ok += 1
        details = obj.get("details")
        if isinstance(details, dict) and all(k in details for k in REQUIRED_DETAIL):
            detail_ok += 1

    return {
        "median_ms": statistics.median(latencies) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "prompt_tokens": prompt_tokens,
        "completion_tokens_median": int(statistics.median(completions)),
        "valid_json": valid_json,
        "correctly_typed": typed_ok,
        "complete_details": detail_ok,
        "runs": runs,
        "sample": sample[:300],
    }


def report(label: str, r: Dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"  latency        {r['median_ms']:8.1f} ms median "
          f"({r['min_ms']:.0f}-{r['max_ms']:.0f} ms, n={r['runs']})")
    print(f"  tokens         prompt={r['prompt_tokens']} "
          f"completion={r['completion_tokens_median']}")
    print(f"  valid JSON     {r['valid_json']}/{r['runs']}")
    print(f"  correct types  {r['correctly_typed']}/{r['runs']}")
    print(f"  full details   {r['complete_details']}/{r['runs']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000",
                    help="vLLM base URL (default: %(default)s)")
    ap.add_argument("--model", default=None,
                    help="Model id; defaults to the server's first served model")
    ap.add_argument("--image", type=Path, default=None,
                    help="Frame to inspect; defaults to one sampled from the video clip")
    ap.add_argument("--video", type=Path, default=Path("IMG_1043.MOV"),
                    help="Video simulator clip used when --image is omitted")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--json", type=Path, default=None,
                    help="Write full results to this path")
    args = ap.parse_args()

    model = args.model
    if not model:
        try:
            data = requests.get(f"{args.url.rstrip('/')}/v1/models", timeout=15).json()
            model = data["data"][0]["id"]
        except Exception as exc:
            raise SystemExit(f"Could not auto-detect model from {args.url}: {exc}")

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Cannot read image: {args.image}")
    else:
        frame = frame_from_video(args.video)

    image_b64, encode_s = encode(frame)

    print(f"model:  {model}")
    print(f"server: {args.url}")
    print(f"frame:  {frame.shape[1]}x{frame.shape[0]} source")
    print(f"\n  resize + JPEG encode  {encode_s * 1000:8.1f} ms (CPU, off the model)")

    # Prefill isolation: max_tokens=1 leaves prefill plus a single decode step.
    warm = measure(args.url, build_payload(model, image_b64, max_tokens=1,
                                           schema=PCB_INSPECTION_RESPONSE_SCHEMA),
                   runs=3)
    print(f"  prefill (warm cache)  {warm['median_ms']:8.1f} ms "
          f"({warm['prompt_tokens']} prompt tokens)")

    # A mirrored frame the server has never seen forces a genuine image prefill.
    novel_b64, _ = encode(cv2.flip(frame, 1))
    cold = measure(args.url, build_payload(model, novel_b64, max_tokens=1,
                                           schema=PCB_INSPECTION_RESPONSE_SCHEMA),
                   runs=1, warm=False)
    print(f"  prefill (novel image) {cold['median_ms']:8.1f} ms")

    # Mirror the shipped call: the schema bounds the response, so the token
    # ceiling is only a backstop against a pathological run.
    structured = measure(args.url, build_payload(
        model, image_b64, max_tokens=STRUCTURED_MAX_TOKENS,
        schema=PCB_INSPECTION_RESPONSE_SCHEMA), runs=args.runs)
    freeform = measure(args.url, build_payload(
        model, image_b64, max_tokens=2048, schema=None), runs=args.runs)

    report("STRUCTURED  (response_format json_schema)", structured)
    report("FREE-FORM   (prompt-only JSON, pre-change behaviour)", freeform)

    decode_ms = structured["median_ms"] - warm["median_ms"]
    per_token = decode_ms / max(structured["completion_tokens_median"], 1)
    print("\n--- where the time goes ---")
    print(f"  decode          {decode_ms:.0f} ms "
          f"({decode_ms / structured['median_ms'] * 100:.0f}% of the call), "
          f"{per_token:.1f} ms/token")
    print(f"  image prefill   {cold['median_ms']:.0f} ms worst case "
          f"({cold['median_ms'] / structured['median_ms'] * 100:.0f}%)")
    print(f"  => this call is decode-bound; token count is the lever, "
          f"not prompt caching.")

    if args.json:
        args.json.write_text(json.dumps({
            "model": model, "url": args.url, "runs": args.runs,
            "encode_ms": encode_s * 1000,
            "prefill_warm": warm, "prefill_novel": cold,
            "structured": structured, "freeform": freeform,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
