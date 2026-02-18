from agents.vlm.client import TaskType, UnifiedVLMClient


def test_vlm_fallback_is_non_detection_and_marks_parse_error():
    client = UnifiedVLMClient.__new__(UnifiedVLMClient)

    result = client._create_fallback_analysis_result(
        task_type=TaskType.CUSTOM,
        raw_response="not-json",
    )

    assert result.detected is False
    assert result.confidence == 0.0
    assert result.should_alert is False
    assert result.details.get("parse_error") is True
