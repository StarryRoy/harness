from datetime import timezone

from agent_harness import (
    CompositeEventSink,
    MetricsEventSink,
    ObservabilityConfig,
    RuntimeEvent,
    RuntimeObserver,
    sanitize_payload,
)


class Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class BrokenSink:
    def emit(self, event):
        raise RuntimeError("logging backend unavailable")


def test_runtime_event_has_stable_fields_and_timezone():
    event = RuntimeEvent(
        "agent.start",
        agent_name="assistant",
        session_id="session",
        trace_id="trace",
        span_id="span",
        status="started",
    )

    assert event.timestamp.tzinfo == timezone.utc
    assert set(event.as_dict()) == {
        "event_type",
        "timestamp",
        "agent_name",
        "session_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "status",
        "duration_ms",
        "metadata",
        "error",
    }


def test_payload_is_redacted_omitted_and_truncated():
    config = ObservabilityConfig(payload_detail="standard", max_payload_chars=8)
    value = sanitize_payload(
        {
            "api_key": "top-secret",
            "messages": ["private prompt"],
            "result": "abcdefghijklmnop",
            "nested": {"password": "guess-me"},
        },
        config,
    )

    assert value["api_key"] == "[REDACTED]"
    assert value["messages"] == "[omitted]"
    assert value["result"].endswith("...[truncated]")
    assert value["nested"]["password"] == "[REDACTED]"


def test_sink_failures_are_isolated_and_metrics_collect_spans():
    recorder = Recorder()
    metrics = MetricsEventSink()
    observer = RuntimeObserver([CompositeEventSink([BrokenSink(), recorder]), metrics])

    with (
        observer.trace(agent_name="assistant", session_id="s"),
        observer.span("model", purpose="agent") as span,
    ):
        span.metadata["token_usage"] = {"input_tokens": 2, "output_tokens": 3}

    assert [event.event_type for event in recorder.events] == [
        "model.start",
        "model.end",
    ]
    snapshot = metrics.snapshot()
    assert snapshot.model_calls == 1
    assert snapshot.model_duration_ms >= 0
    assert snapshot.input_tokens == 2
    assert snapshot.output_tokens == 3
