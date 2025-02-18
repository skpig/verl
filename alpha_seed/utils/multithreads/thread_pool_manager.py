from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time


class ThreadPoolManager:

    def __init__(self, max_workers=None):
        """
        Initialize the thread pool manager.
        :param max_workers: Maximum number of worker threads. Defaults to min(32, os.cpu_count() + 4).
        """
        self.max_workers = max_workers if max_workers else min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.futures = []

    def submit_task(self, func, *args, **kwargs):
        """
        Submit a task to the thread pool.
        :param func: The task function to execute.
        :param args: Positional arguments for the task function.
        :param kwargs: Keyword arguments for the task function.
        """
        future = self.executor.submit(func, *args, **kwargs)
        self.futures.append(future)

    def wait_for_completion(self):
        """
        Wait for all tasks to complete and handle exceptions.
        """
        for future in as_completed(self.futures):
            try:
                future.result()  # Get task result, catch exceptions
            except Exception as e:
                print(f"Task failed with error: {e}")
                self.executor.shutdown(wait=False)
                raise RuntimeError("A worker thread encountered an error.") from e

    def shutdown(self):
        """
        Manually shut down the thread pool.
        """
        self.executor.shutdown(wait=True)
        print("Thread pool shutdown complete.")
