"""Tests for the two-model router split.

The system serves a vision VLM and a text/agent model from two separate vLLM
pods. These pin the wiring that keeps them independent, and the fallback that
lets a single-pod deployment keep working.
"""

import pytest

from router import llm_router
from router.llm_router import ROLE_AGENT, ROLE_VISION, get_router, reset_routers
from router.resilience import get_concurrency_limiter


VISION_URL = "http://vision.invalid:8000"
AGENT_URL = "http://agent.invalid:8000"


@pytest.fixture(autouse=True)
def _isolated_routers(monkeypatch):
    """Give each test a clean router cache and no availability probing.

    Router construction performs a health check; these tests are about
    configuration wiring, not reachability.
    """
    monkeypatch.setattr(
        llm_router.AgentLLMRouter, "_check_availability", lambda self: False
    )
    reset_routers()
    yield
    reset_routers()


def _configure(monkeypatch, *, agent_url=AGENT_URL, agent_model="LiquidAI/LFM2.5-2.6B"):
    monkeypatch.setenv("VLLM_URL", VISION_URL)
    monkeypatch.setenv("VISION_MODEL", "LiquidAI/LFM2.5-VL-1.6B-PCB-Inspect")
    for key, value in (("AGENT_LLM_URL", agent_url), ("AGENT_MODEL", agent_model)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_roles_resolve_to_separate_routers(monkeypatch):
    _configure(monkeypatch)
    vision = get_router(ROLE_VISION)
    agent = get_router(ROLE_AGENT)

    assert vision is not agent
    assert vision.role == ROLE_VISION
    assert agent.role == ROLE_AGENT


def test_default_role_is_vision(monkeypatch):
    """Callers not updated for the split must keep hitting the vision model."""
    _configure(monkeypatch)
    assert get_router() is get_router(ROLE_VISION)


def test_each_role_gets_its_own_endpoint_and_model(monkeypatch):
    _configure(monkeypatch)
    vision = get_router(ROLE_VISION).get_config()
    agent = get_router(ROLE_AGENT).get_config()

    assert vision.url == VISION_URL
    assert agent.url == AGENT_URL
    assert vision.model == "LiquidAI/LFM2.5-VL-1.6B-PCB-Inspect"
    assert agent.model == "LiquidAI/LFM2.5-2.6B"


def test_only_the_vision_role_advertises_vision(monkeypatch):
    _configure(monkeypatch)
    assert get_router(ROLE_VISION).get_config().supports_vision is True
    assert get_router(ROLE_AGENT).get_config().supports_vision is False
    # Both drive tools; the agent model is the one that actually will.
    assert get_router(ROLE_AGENT).get_config().supports_tools is True


def test_roles_do_not_share_a_concurrency_limiter(monkeypatch):
    """A multi-second vision inference must not consume the agent's slots."""
    _configure(monkeypatch)
    vision_name = get_router(ROLE_VISION).get_config().name
    agent_name = get_router(ROLE_AGENT).get_config().name

    assert vision_name != agent_name
    assert get_concurrency_limiter(vision_name) is not get_concurrency_limiter(agent_name)


def test_routers_are_cached_per_role(monkeypatch):
    _configure(monkeypatch)
    assert get_router(ROLE_AGENT) is get_router(ROLE_AGENT)


def test_agent_falls_back_to_the_vision_endpoint_when_unset(monkeypatch):
    """Single-pod deployments (agentServer.enabled=false) must still work."""
    _configure(monkeypatch, agent_url=None, agent_model=None)
    agent = get_router(ROLE_AGENT).get_config()

    assert agent.url == VISION_URL
    assert agent.model == "LiquidAI/LFM2.5-VL-1.6B-PCB-Inspect"
    # Still a distinct limiter key even when sharing an endpoint.
    assert agent.name != get_router(ROLE_VISION).get_config().name


def test_unknown_role_is_rejected(monkeypatch):
    _configure(monkeypatch)
    with pytest.raises(ValueError):
        get_router("embedding")
