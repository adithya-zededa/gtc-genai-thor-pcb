from agents.core.state import AgentMemory
from dataclasses import asdict


def test_agent_memory_counts_and_summary():
    memory = AgentMemory(max_events=5, summary_window=3)

    memory.add_event(
        {
            "timestamp": "2025-11-17T10:00:00",
            "detected": False,
            "primary_label": "no_detection",
            "confidence": 0.1,
            "pcb_stable": None,
            "should_alert": False,
            "source": "analysis",
        }
    )

    memory.add_event(
        {
            "timestamp": "2025-11-17T10:01:00",
            "detected": True,
            "primary_label": "pcb_unstable",
            "confidence": 0.85,
            "pcb_stable": False,
            "should_alert": True,
            "source": "analysis",
        }
    )

    memory.add_event(
        {
            "timestamp": "2025-11-17T10:02:00",
            "detected": True,
            "primary_label": "pcb_stable",
            "confidence": 0.65,
            "pcb_stable": True,
            "should_alert": False,
            "source": "agent_ssim_guard",
        }
    )

    snapshot = asdict(memory.snapshot())

    assert snapshot["counts"]["total"] == 3
    assert snapshot["counts"]["detections"] == 2
    assert snapshot["counts"]["unstable"] == 1
    assert snapshot["counts"]["alerts"] == 1

    summary = memory.summarise()

    assert "Processed 3" in summary
    assert "Detections: 2" in summary
    assert "Alerts raised: 1" in summary
    assert "Last event" in summary


def test_agent_memory_resize_preserves_recent_entries():
    memory = AgentMemory(max_events=4, summary_window=4)

    for idx in range(6):
        memory.add_event(
            {
                "timestamp": f"2025-11-17T10:0{idx}:00",
                "detected": idx % 2 == 0,
                "pcb_stable": False if idx % 3 == 0 else None,
                "should_alert": idx % 2 == 0,
                "source": "analysis",
            }
        )

    snapshot_before = asdict(memory.snapshot())
    assert snapshot_before["counts"]["total"] == 4

    memory.resize(2, 1)

    snapshot_after = asdict(memory.snapshot())
    assert snapshot_after["counts"]["total"] == 1
    assert snapshot_after["last_event"]["timestamp"].endswith("10:05:00")

    summary = memory.summarise()
    assert summary.startswith("Processed 1")
