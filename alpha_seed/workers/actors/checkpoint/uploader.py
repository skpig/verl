import os
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import hdfs_io
import asyncio
import shutil
from omnistore.utilities.io.bfile import is_local_path


@ray.remote(num_gpus=0, num_cpus=1)
class CkptGlobalUploader:

    name = "checkpoint_global_uploader"

    def __init__(self, tracker_role, ckpt_version, default_local_dir, default_remote_dir, upload_retry_count):
        # use tracker_role to specify who is responsible for updating the tracker file
        self.upload_shard_future_map = {}
        self.upload_shard_task_map = {}
        self.callback_condition_map = {}
        self.async_resource_lock = asyncio.Lock()
        self.tracker_role = tracker_role
        print(f'tracker role {self.tracker_role}')
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

    async def start_uploading(self, role, global_step=0):
        # only rank 0 should call this function
        async with self.async_resource_lock:
            if global_step not in self.upload_shard_future_map:
                self.upload_shard_future_map[global_step] = {}

            await asyncio.to_thread(self.prepare_remote_paths,
                                    {item[2] for item in self.upload_shard_task_map[global_step][role]})
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
        await self.write_tracker(global_step)

    def register_callback(self, role, global_step):
        if global_step not in self.callback_condition_map:
            self.callback_condition_map[global_step] = {}
        if self.callback_condition_map[global_step].get(role) is None:
            self.callback_condition_map[global_step][role] = {}
        self.callback_condition_map[global_step][role] = {
            'cond': asyncio.Condition(),
            'called': False,
        }

    async def callback(self, role, global_step):
        if global_step not in self.callback_condition_map or role not in self.callback_condition_map[global_step]:
            print(f'callback role {role} global step {global_step} condition not found')
            return
        async with self.callback_condition_map[global_step][role]['cond']:
            self.callback_condition_map[global_step][role]['cond'].notify_all()
            self.callback_condition_map[global_step][role]['called'] = True

    @staticmethod
    def prepare_remote_paths(remote_path_set):
        # only rank 0 should call this function
        for remote_path in remote_path_set:
            hdfs_io.makedirs(remote_path, exist_ok=True)

    async def final_wait_all_steps(self):
        results = []
        async with self.async_resource_lock:
            upload_shard_future_map_keys = list(self.upload_shard_future_map.keys())
        for global_step in upload_shard_future_map_keys:
            results.append(self.wait_all(global_step, need_clear=False))
        return all(results)

    async def wait_all(self, global_step, need_clear=True):
        # wait futures
        results = []
        future_map_value = None
        async with self.async_resource_lock:
            if global_step in self.upload_shard_future_map:
                future_map_value = self.upload_shard_future_map[global_step]
        if future_map_value is not None:
            for future_list_by_role in future_map_value.values():
                tmp_results = await asyncio.gather(*future_list_by_role)
                results.append(tmp_results)

        # wait callbacks
        async def _wait_multiple_callbacks():
            async with self.async_resource_lock:
                if global_step not in self.callback_condition_map:
                    return
                else:
                    callback_condition_map_value = self.callback_condition_map[global_step]
            for value in callback_condition_map_value.values():
                if value['called']:
                    continue
                async with value['cond']:
                    await value['cond'].wait()

        await _wait_multiple_callbacks()

        if need_clear:
            async with self.async_resource_lock:
                self.clear_futures(global_step)
                self.clear_tasks(global_step)
                self.clear_callback_conditions(global_step)
        return all(results)

    async def wait_by_role(self, role, global_step):
        # wait futures
        results = []
        futures = None
        async with self.async_resource_lock:
            if global_step in self.upload_shard_future_map and role in self.upload_shard_future_map[global_step]:
                futures = self.upload_shard_future_map[global_step][role]
        if futures is not None:
            results = await asyncio.gather(*futures)

        # wait callback
        async def _wait_single_callback():
            async with self.async_resource_lock:
                if global_step not in self.callback_condition_map:
                    return
                else:
                    condition_value_by_role = self.callback_condition_map[global_step].get(role, None)
            if condition_value_by_role is None or condition_value_by_role['called'] is True:
                return
            async with condition_value_by_role['cond']:
                await condition_value_by_role['cond'].wait()

        await _wait_single_callback()
        return all(results)

    def clear_futures(self, global_step):
        if global_step not in self.upload_shard_future_map:
            return
        del self.upload_shard_future_map[global_step]

    def clear_tasks(self, global_step):
        if global_step not in self.upload_shard_task_map:
            return
        del self.upload_shard_task_map[global_step]

    def clear_callback_conditions(self, global_step):
        if global_step not in self.callback_condition_map:
            return
        del self.callback_condition_map[global_step]

    async def write_tracker(self, global_step):
        wait_result = await self.wait_all(global_step)
        if not wait_result:
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
        print(f"write_tracker: write {str(global_step)} to {local_latest_checkpointed_iteration} success")

        if not is_local_path(self.remote_checkpoint_folder):
            await asyncio.to_thread(hdfs_io.hcopy, local_latest_checkpointed_iteration, self.remote_checkpoint_folder)

        # mark a checkpoint version for future checkpoint format change and compatibility
        local_ckpt_version = os.path.join(self.local_checkpoint_folder, 'checkpoint_version.txt')
        with open(local_ckpt_version, 'w') as f:
            f.write(self.ckpt_version)
        print(f"write_tracker: write {self.ckpt_version} to {local_ckpt_version} success")

        if not is_local_path(self.remote_checkpoint_folder):
            await asyncio.to_thread(hdfs_io.hcopy, local_ckpt_version, self.remote_checkpoint_folder)


@ray.remote
def upload_ckpt_with_retry(local_path, remote_path, upload_retry_count):
    local_path = os.path.abspath(local_path)

    result = False
    for current_retry_count in range(upload_retry_count):
        result = upload_ckpt(local_path, remote_path)
        if result:
            break
        print(
            f'Uploading checkpoint from {local_path} to {remote_path} failed, current retry count '
            f'{current_retry_count}, max retry count {upload_retry_count}',
            flush=True)
    return result


def upload_ckpt(local_path, remote_path):
    if is_local_path(remote_path):
        try:
            shutil.copy(local_path, remote_path)
        except shutil.SameFileError:
            print(f"upload_ckpt: local_path={local_path} is same as remote_path={remote_path}, skipping upload")
        except Exception:
            return False
        return True

    try:
        hdfs_io.copy(src=local_path, dst=remote_path)
    except Exception:
        return False
    if os.path.exists(local_path):
        if os.path.isfile(local_path):
            try:
                os.remove(local_path)
            except Exception as e:
                print(f"remove {local_path} failed. error: ", e)
        else:
            shutil.rmtree(local_path, ignore_errors=True)

    return True
