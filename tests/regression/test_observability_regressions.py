from agent_harness import MetricsEventSink, RuntimeObserver


class BrokenSink:
    def emit(self, event):
        raise RuntimeError("sink down")


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_observer_sink_failure_cannot_break_trace():
    recording = RecordingSink()
    metrics = MetricsEventSink()
    observer = RuntimeObserver([BrokenSink(), recording, metrics])

    with observer.trace(agent_name="regression", session_id="session"):
        observer.emit("agent.start", status="started")
        observer.emit("agent.end", status="success", duration_ms=1)

    assert [event.event_type for event in recording.events] == [
        "agent.start",
        "agent.end",
    ]
    assert metrics.snapshot().agent_calls == 1
