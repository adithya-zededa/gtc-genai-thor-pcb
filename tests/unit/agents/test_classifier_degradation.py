import time

from agents.classifiers.llm_classifier import LLMIntentClassifier


def test_classifier_returns_safe_general_on_llm_error(monkeypatch):
    classifier = LLMIntentClassifier(timeout=1)

    def _raise(_message: str):
        raise RuntimeError("backend down")

    monkeypatch.setattr(classifier, "_call_llm", _raise)

    result = classifier.classify("inspect the board")
    metrics = classifier.get_health_metrics()

    assert result.domain == "general"
    assert result.tool is None
    assert result.source == "llm_error"
    assert metrics["llm_failures"] >= 1
    assert metrics["fallback_general_returns"] >= 1


def test_classifier_circuit_open_short_circuits_to_general():
    classifier = LLMIntentClassifier(timeout=1)
    classifier._consecutive_failures = 3
    classifier._circuit_open_until = time.time() + 10

    result = classifier.classify("start monitoring")
    metrics = classifier.get_health_metrics()

    assert result.domain == "general"
    assert result.source == "circuit_breaker"
    assert metrics["circuit_open_returns"] >= 1
