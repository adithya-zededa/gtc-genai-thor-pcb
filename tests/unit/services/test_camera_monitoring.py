from types import SimpleNamespace

import pytest

from services.core.monitoring import StreamlinedMonitoringService as CameraMonitoringService


class FakePublisher:
    """Minimal publisher stub that tracks subscription activity."""

    def __init__(self, running: bool = True) -> None:
        self.is_running = running
        self.subscribed = set()
        self.unsubscribe_calls = []
        self.start_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        self.is_running = True
        return True

    def subscribe(
        self, subscriber_id: str, callback=None
    ) -> bool:  # pragma: no cover - callback unused
        self.subscribed.add(subscriber_id)
        return True

    def unsubscribe(self, subscriber_id: str) -> None:
        self.unsubscribe_calls.append(subscriber_id)
        self.subscribed.discard(subscriber_id)

    def get_frame(
        self, _subscriber_id: str, timeout: float = 0.5
    ):  # pragma: no cover - monitoring loop patched
        return None

    def get_latest_frame(self):  # pragma: no cover
        return None


def _mark_initialized(service, agent, publisher=None):
    """Helper: set agent + initialized flag so tests skip real init."""
    service.agent = agent
    if publisher is not None:
        service.publisher = publisher
    # Directly set the internal flag via the core lock
    with service._core_lock:
        service._initialized = True


@pytest.fixture
def dummy_agent():
    ollama_client = SimpleNamespace(test_connection=lambda: True)
    return SimpleNamespace(
        config={
            "camera": {"capture_interval": 1.5},
            "advanced": {"max_concurrent_analyses": 2},
        },
        ollama_client=ollama_client,
        process_detection=lambda event: False,
    )


def test_apply_configuration_settings_updates_runtime_parameters():
    service = CameraMonitoringService(auto_start_publisher=False)
    config = {
        "camera": {
            "capture_interval": 2.5,
        },
        "advanced": {
            "max_concurrent_analyses": 3,
            "max_pending_analyses": 9,
        },
    }

    service._apply_configuration_settings(
        config
    )  # pylint: disable=protected-access

    assert service.capture_interval == pytest.approx(2.5)
    assert service.analysis_workers == 3
    assert service.max_pending_analyses == 9


def test_refresh_configuration_uses_agent_defaults(dummy_agent):
    service = CameraMonitoringService(auto_start_publisher=False)
    service.agent = dummy_agent

    service.refresh_configuration()

    assert service.capture_interval == pytest.approx(1.5)
    assert service.analysis_workers == 2


def test_start_and_stop_monitoring_manage_subscriptions(
    monkeypatch, dummy_agent
):
    fake_publisher = FakePublisher(running=True)
    service = CameraMonitoringService(
        publisher_getter=lambda: fake_publisher, auto_start_publisher=False
    )
    _mark_initialized(service, dummy_agent, fake_publisher)

    # Bypass LLM-based scope validation (no inference server in unit tests)
    monkeypatch.setattr(
        "agents.core.monitoring_loop.MonitoringLoop.validate_instruction_scope",
        classmethod(lambda cls, instruction: None),
    )

    assert service.start_monitoring() is True

    service.stop_monitoring()

    assert service.is_monitoring is False
    assert service.subscriber_id in fake_publisher.unsubscribe_calls


def test_start_monitoring_autostarts_publisher_when_allowed(
    monkeypatch, dummy_agent
):
    fake_publisher = FakePublisher(running=False)
    service = CameraMonitoringService(
        publisher_getter=lambda: fake_publisher, auto_start_publisher=True
    )
    _mark_initialized(service, dummy_agent, fake_publisher)

    monkeypatch.setattr(
        "agents.core.monitoring_loop.MonitoringLoop.validate_instruction_scope",
        classmethod(lambda cls, instruction: None),
    )

    assert service.start_monitoring() is True
    assert fake_publisher.start_calls >= 1
    service.stop_monitoring()


def test_start_monitoring_explicit_proactive_mode(monkeypatch, dummy_agent):
    fake_publisher = FakePublisher(running=True)
    service = CameraMonitoringService(
        publisher_getter=lambda: fake_publisher,
        auto_start_publisher=False,
    )
    _mark_initialized(service, dummy_agent, fake_publisher)

    called = {}

    def _fake_start_proactive(instruction: str, config=None):
        called["instruction"] = instruction
        called["config"] = config
        return True

    monkeypatch.setattr(service, "start_proactive_monitoring", _fake_start_proactive)

    assert service.start_monitoring(mode="proactive", instruction="Inspect PCB defects") is True
    assert called["instruction"] == "Inspect PCB defects"


def test_start_monitoring_auto_mode_resolves_to_proactive(dummy_agent):
    """'auto' is accepted for compatibility but always resolves to proactive."""
    fake_publisher = FakePublisher(running=True)
    service = CameraMonitoringService(
        publisher_getter=lambda: fake_publisher,
        auto_start_publisher=False,
    )
    _mark_initialized(service, dummy_agent, fake_publisher)

    called = {}

    def _fake_start_proactive(instruction: str, config=None):
        called["started"] = True
        return True

    # Use monkeypatch-style override
    service.start_proactive_monitoring = _fake_start_proactive  # type: ignore[assignment]

    assert service.start_monitoring(mode="auto", instruction="") is True
    assert called.get("started") is True


def test_is_ready_reflects_initialization_state():
    service = CameraMonitoringService(auto_start_publisher=False)
    assert service.is_ready is False

    with service._core_lock:
        service._initialized = True
    assert service.is_ready is True


def test_get_active_monitoring_mode_returns_idle_by_default():
    service = CameraMonitoringService(auto_start_publisher=False)
    assert service.get_active_monitoring_mode() == "idle"


def test_dependency_injection_repos():
    """Injected repos are used instead of lazy imports."""
    fake_repo = SimpleNamespace(create=lambda **kw: 42)
    fake_inspection = SimpleNamespace(create=lambda **kw: 1)
    emitted = []

    service = CameraMonitoringService(
        auto_start_publisher=False,
        detection_repo=fake_repo,
        inspection_repo=fake_inspection,
        socketio_emitter=lambda event, data: emitted.append((event, data)),
    )

    assert service._get_detection_repo() is fake_repo
    assert service._get_inspection_repo() is fake_inspection
    assert service._get_socketio_emit() is not None
