from types import SimpleNamespace

import pytest

from agent_runtime.monitoring import StreamlinedMonitoringService as CameraMonitoringService


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
    service.agent = dummy_agent
    service._monitoring_loop = lambda: None  # type: ignore[attr-defined]
    # type: ignore[attr-defined]
    service._ensure_analysis_executor = lambda: None

    assert service.start_monitoring() is True
    assert service.is_monitoring is True
    assert service.subscriber_id in fake_publisher.subscribed

    service.stop_monitoring()

    assert service.is_monitoring is False
    assert service.subscriber_id in fake_publisher.unsubscribe_calls


def test_start_monitoring_autostarts_publisher_when_allowed(dummy_agent):
    fake_publisher = FakePublisher(running=False)
    service = CameraMonitoringService(
        publisher_getter=lambda: fake_publisher, auto_start_publisher=True
    )
    service.agent = dummy_agent
    service._monitoring_loop = lambda: None  # type: ignore[attr-defined]
    # type: ignore[attr-defined]
    service._ensure_analysis_executor = lambda: None

    assert service.start_monitoring() is True
    assert fake_publisher.start_calls == 1
    service.stop_monitoring()
