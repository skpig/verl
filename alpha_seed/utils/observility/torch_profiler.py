import logging
import os

import torch
import ray

from mono_rl.utils.debug import get_profiler_context


def get_profiler_context_wrapped(filename,
                                 profile_on_ranks,
                                 upload_to_mlx,
                                 enable,
                                 wait=1,
                                 warmup=1,
                                 active=2,
                                 repeat=1):

    default_hdfs_dir = os.environ.get('PROFILE_HDFS_DIR')

    try:
        from mono_rl.utils.debug import MerlinLineageUploader

        # adapt for torchrun
        if ray.is_initialized():
            actor_name = ray.get_runtime_context().get_actor_name()
        else:
            actor_name = 'local'

        merlin_lineage_uploader = MerlinLineageUploader(
            rank=torch.distributed.get_rank(),
            profile_max_preview_rank=0,
            asset_type="perfetto",
            actor_name=actor_name,
        )
        profiler_context = get_profiler_context(filename=filename,
                                                profile_on_ranks=profile_on_ranks,
                                                default_hdfs_dir=default_hdfs_dir,
                                                upload_to_mlx=upload_to_mlx,
                                                enable=enable,
                                                wait=wait,
                                                warmup=warmup,
                                                active=active,
                                                repeat=repeat,
                                                profiler_uploader=merlin_lineage_uploader)
    except ImportError:
        profiler_context = get_profiler_context(filename=filename,
                                                profile_on_ranks=profile_on_ranks,
                                                default_hdfs_dir=default_hdfs_dir,
                                                upload_to_mlx=upload_to_mlx,
                                                enable=enable,
                                                wait=wait,
                                                warmup=warmup,
                                                active=active,
                                                repeat=repeat)

    return profiler_context


def profile_step(p, step=None):
    try:
        from mono_rl.utils.debug import VerlProfiler
        if type(p) is VerlProfiler and step is not None:
            p.set_step(step)
    except ImportError:
        pass
    p.step()
