import os
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import hdfs_io


@ray.remote(num_gpus=0, num_cpus=1)
class CkptGlobalUploader:

    name = "checkpoint_global_uploader"

    def __init__(self, tracker_role, ckpt_version, default_local_dir, default_remote_dir, upload_retry_count):
        # use tracker_role to specify who is responsible for updating the tracker file
        self.upload_shard_future_map = {}
        self.upload_shard_task_map = {}
        self.tracker_role = tracker_role
        self.ckpt_version = ckpt_version
        self.local_checkpoint_folder = os.path.join(default_local_dir, 'checkpoints')
        os.makedirs(self.local_checkpoint_folder, exist_ok=True)
        self.remote_checkpoint_folder = os.path.join(default_remote_dir, 'checkpoints')
        self.upload_retry_count = upload_retry_count

    def register_upload_task(self, role, global_step, node_id, local_path, remote_path):
        if global_step not in self.upload_shard_task_map:
            self.upload_shard_task_map[global_step] = {}
        if self.upload_shard_task_map[global_step].get(role) is None:
            self.upload_shard_task_map[global_step][role] = []
        self.upload_shard_task_map[global_step][role].append((node_id, local_path, remote_path))

    def start_uploading(self, role, global_step=0):
        # only rank 0 should call this function
        print(f'checkpoint global uploader start to upload role {role} global step {global_step}', flush=True)
        if global_step not in self.upload_shard_future_map:
            self.upload_shard_future_map[global_step] = {}

        self.prepare_remote_paths({item[2] for item in self.upload_shard_task_map[global_step][role]})
        for node_id, local_path, remote_path in self.upload_shard_task_map[global_step][role]:
            upload_shard_future = upload_ckpt_with_retry.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )).remote(local_path, remote_path, self.upload_retry_count)
            if self.upload_shard_future_map[global_step].get(role) is None:
                self.upload_shard_future_map[global_step][role] = []
            self.upload_shard_future_map[global_step][role].append(upload_shard_future)

        if role != self.tracker_role:
            return
        self.write_tracker(global_step)

    @staticmethod
    def prepare_remote_paths(remote_path_set):
        # only rank 0 should call this function
        for remote_path in remote_path_set:
            hdfs_io.makedirs(remote_path, exist_ok=True)

    def final_wait_all_steps(self):
        results = []
        for global_step in self.upload_shard_future_map.keys():
            results.append(self.wait_all(global_step, need_clear=False))
        return all(results)

    def wait_all(self, global_step, need_clear=True):
        results = []
        if global_step not in self.upload_shard_future_map:
            return True
        for role in self.upload_shard_future_map[global_step].keys():
            results.append(all(ray.get(self.upload_shard_future_map[global_step][role])))
        if need_clear:
            self.clear_futures(global_step)
            self.clear_tasks(global_step)
        return all(results)

    def wait_by_role(self, role, global_step):
        if global_step not in self.upload_shard_future_map:
            return True
        results = ray.get(self.upload_shard_future_map[global_step][role])
        return all(results)

    def clear_futures(self, global_step):
        if global_step not in self.upload_shard_future_map:
            return
        del self.upload_shard_future_map[global_step]

    def clear_tasks(self, global_step):
        if global_step not in self.upload_shard_task_map:
            return
        del self.upload_shard_task_map[global_step]

    def write_tracker(self, global_step):
        if not self.wait_all(global_step):
            print(
                f'checkpoint global uploader wait for step {global_step} failed, upload some checkpoint files failed, '
                'will not update latest_checkpointed_iteration.txt')
            return
        print(
            f'checkpoint global uploader wait for step {global_step} done, will update latest_checkpointed_iteration.txt',
            flush=True)
        local_latest_checkpointed_iteration = os.path.join(self.local_checkpoint_folder,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(global_step))
        hdfs_io.hput(local_latest_checkpointed_iteration, self.remote_checkpoint_folder)

        # mark a checkpoint version for future checkpoint format change and compatibility
        local_ckpt_version = os.path.join(self.local_checkpoint_folder, 'checkpoint_version.txt')
        with open(local_ckpt_version, 'w') as f:
            f.write(self.ckpt_version)
        hdfs_io.hput(local_ckpt_version, self.remote_checkpoint_folder)


@ray.remote
def upload_ckpt_with_retry(local_path, remote_path, upload_retry_count):
    local_path = os.path.abspath(local_path)
    print(f'Start uploading checkpoint with retry from {local_path} to {remote_path}', flush=True)

    result = False
    for current_retry_count in range(upload_retry_count):
        result = upload_ckpt(local_path, remote_path)
        if result:
            break
        print(
            f'Uploading checkpoint from {local_path} to {remote_path} failed, current retry count '
            f'{current_retry_count}, max retry count {upload_retry_count}',
            flush=True)
    print(f'Finish uploading checkpoint with retry from {local_path} to {remote_path}, final result: {result}',
          flush=True)
    return result


def upload_ckpt(local_path, remote_path):
    try:
        hdfs_io.copy(src=local_path, dst=remote_path)
    except Exception:
        return False
    return True
