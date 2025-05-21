from typing import List
import ray

from alpha_seed.utils.profile.timeline import Tracer, export_chrome_trace
from alpha_seed.workers.streaming_service.rollout_request_manager import get_all_request_manager_actors


class RolloutQueryTimeline:

    def __init__(self, config):
        self.config = config
        self.save_path = None

    def step(self, global_step):
        if global_step == self.config.to_step:
            # export at TaskRunner process
            spans: List[dict] = Tracer.merge_all()

            for req_mgr in get_all_request_manager_actors():
                request_spans = ray.get(req_mgr.dump_request_trace.remote())
                spans.extend(request_spans)
            self.save_path = export_chrome_trace('query_trace.json.gz', spans)

        if self.save_path is not None and self.config.upload_to_mlx and global_step == self.config.to_step + 1:
            # delay upload to next step to make sure file be flushed properly
            import subprocess
            ret = subprocess.call([f"/opt/tiger/mlx_deploy/bin/mlx asset upload {self.save_path}"], shell=True)
            if ret != 0:
                print("upload profiler trace fail. please see the log around")
