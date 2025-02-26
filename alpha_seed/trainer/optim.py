import warnings


def get_optimizer_from_config(param_groups, optimizer_config):
    type = optimizer_config.type
    lr = optimizer_config.lr
    betas = optimizer_config.get("betas", (0.9, 0.95))
    eps = optimizer_config.get("eps", 1e-08)
    weight_decay = optimizer_config.get("weight_decay", 0.1)
    force_bfloat16_state = optimizer_config.get("force_bfloat16_state", False)
    if type == "adam":
        import torch
        warnings.warn(f"Using torch.optim.AdamW. force_bfloat16_state is not supported.")
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        try:
            from byted_optimizer.alphaseed import get_byted_optimizer_from_alphaseed_config
            alpha_seed_config = {
                "type": type,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "force_bfloat16_state": force_bfloat16_state
            }
            optimizer = get_byted_optimizer_from_alphaseed_config(param_groups, alpha_seed_config)
        except ImportError:
            warnings.warn(
                f"Cannot find byted_optimizer. Try to use apex.optimizers instead. force_bfloat16_state is not supported."
            )
            from apex import optimizers
            if type == "adamw":
                apex_optimizer = optimizers.FusedAdam
            elif type == "lamb":
                apex_optimizer = optimizers.FusedLAMB
            else:
                raise NotImplementedError(f"Optimizer type: {type} is not supported by Apex.")
            optimizer = apex_optimizer(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return optimizer
