from megatron import get_args
from megatron.optimizer import get_megatron_optimizer
from megatron.training import get_optimizer_param_scheduler

from megatron.core import parallel_state as mpu


def configure_optimizers(models,
                         train_iters: int,
                         lr_warmup_iters: int,
                         lr: float,
                         adam_betas=(0.9, 0.999),
                         adam_eps=1e-8,
                         weight_decay=1e-2):
    # models should consist of model chunks
    args = get_args()
    args.optimizer = 'adam'
    assert args.optimizer == 'adam'
    args.train_iters = train_iters
    args.lr = lr
    args.adam_beta1 = adam_betas[0]
    args.adam_beta2 = adam_betas[1]
    args.adam_eps = adam_eps
    args.weight_decay = weight_decay
    args.lr_warmup_iters = lr_warmup_iters
    args.lr_decay_style = 'constant'
    args.lr_decay_steps = (train_iters - lr_warmup_iters)
    args.min_lr = args.lr
    # do not start from 0, otherwise the first step is wasted.
    args.warmup_start_lr = args.lr * 0.1

    optimizer = get_megatron_optimizer(models, None, None, 1.0)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer, lr_update_by_global_bsz=False)

    assert args.load is None

    # We only support local DDP with multiple micro-batches.
    if len(models) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    optimizers = [optimizer]
    lr_schedulers = [opt_param_scheduler]
    return optimizers, lr_schedulers
