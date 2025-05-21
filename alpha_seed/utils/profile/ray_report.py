from typing import List

import ray


@ray.remote(num_cpus=0.1)
class TraceReporter:

    def __init__(self):
        self.event_spans: List[dict] = []

    def report(self, spans: List[dict]):
        self.event_spans.extend(spans)

    def get_events(self) -> List[dict]:
        return self.event_spans
