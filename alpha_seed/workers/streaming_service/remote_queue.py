import queue
import threading

import ray.util.queue
import ray.util


class QueueIter:

    def __init__(self, queue: ray.util.queue.Queue) -> None:
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()


class RemoteQueue:

    def __init__(self, max_prefetch: int = 8):
        self.max_prefetch = max_prefetch
        self.generator_queue = queue.Queue()  # 放连续的generator的queue
        scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        actor_options = {'scheduling_strategy': scheduling_strategy}
        self.queue = ray.util.queue.Queue(max_prefetch, actor_options)
        self._t = threading.Thread(target=self._produce_loop, name="RemoteQueueProducer", daemon=True)
        self._t.start()

    def produce(self, generator):
        self.generator_queue.put(generator)

    def _produce_loop(self):
        gen = self.generator_queue.get()
        while True:
            try:
                item = next(gen)
                self.queue.put(item)
            except StopIteration:
                gen = self.generator_queue.get()

    def consumer(self) -> QueueIter:
        return QueueIter(self.queue)
