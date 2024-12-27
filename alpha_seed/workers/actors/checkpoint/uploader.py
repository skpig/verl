import os
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import hdfs_io


@ray.remote(num_gpus=0, num_cpus=1)
class CkptGlobalUploader:

    name = "checkpoint_global_uploader"

    def __init__(self, use_critic, ckpt_version, default_local_dir, default_remote_dir):
        self.upload_shard_future_map = {}
        self.upload_shard_task_map = {}
        self.use_critic = use_critic
        self.ckpt_version = ckpt_version
        self.local_checkpoint_folder = os.path.join(default_local_dir, 'checkpoints')
        self.remote_checkpoint_folder = os.path.join(default_remote_dir, 'checkpoints')

    def register_upload_task(self, role, global_step, node_id, local_path, remote_path):
        if global_step not in self.upload_shard_task_map:
            self.upload_shard_task_map[global_step] = {'actor': [], 'critic': [], 'default': []}
        self.upload_shard_task_map[global_step][role].append((node_id, local_path, remote_path))

    def start_uploading(self, role, global_step=0):
        # only rank 0 should call this function
        print(f'checkpoint global uploader start to upload role {role} global step {global_step}', flush=True)
        if global_step not in self.upload_shard_future_map:
            self.upload_shard_future_map[global_step] = {'actor': [], 'critic': [], 'default': []}

        self.prepare_remote_paths({item[2] for item in self.upload_shard_task_map[global_step][role]})
        for node_id, local_path, remote_path in self.upload_shard_task_map[global_step][role]:
            upload_shard_future = upload_ckpt.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )).remote(local_path, remote_path)
            self.upload_shard_future_map[global_step][role].append(upload_shard_future)

        if (self.use_critic and role != 'critic') or role == 'default':
            return
        self.write_tracker(global_step)

    @staticmethod
    def prepare_remote_paths(remote_path_set):
        # only rank 0 should call this function
        for remote_path in remote_path_set:
            hdfs_io.makedirs(remote_path, exist_ok=True)

    def wait_all(self, global_step, need_clear=True):
        for role in ['actor', 'critic', 'default']:
            self.wait_by_role(role, global_step)
        if need_clear:
            self.clear_futures(global_step)
            self.clear_tasks(global_step)

    def wait_by_role(self, role, global_step):
        if global_step not in self.upload_shard_future_map:
            return
        ray.get(self.upload_shard_future_map[global_step][role])

    def clear_futures(self, global_step):
        if global_step not in self.upload_shard_future_map:
            return
        del self.upload_shard_future_map[global_step]

    def clear_tasks(self, global_step):
        if global_step not in self.upload_shard_task_map:
            return
        del self.upload_shard_task_map[global_step]

    def write_tracker(self, global_step):
        self.wait_all(global_step)
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
def upload_ckpt(local_path, remote_path):
    local_path = os.path.abspath(local_path)
    print(f'Start uploading checkpoint from {local_path} to {remote_path}', flush=True)
    hdfs_io.copy(src=local_path, dst=remote_path)
    print(f'Finish uploading checkpoint from {local_path} to {remote_path}', flush=True)
