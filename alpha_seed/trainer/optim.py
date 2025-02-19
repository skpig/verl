import warnings


def get_optimizer_from_config(param_groups, optimizer_config):
    type = optimizer_config.type
    lr = optimizer_config.lr
    betas = optimizer_config.get("betas", (0.9, 0.95))
    eps = optimizer_config.get("eps", 1e-08)
    weight_decay = optimizer_config.get("weight_decay", 0.1)
    force_bfloat16_state = optimizer_config.get("force_bfloat16_state", False)
    try:
        from byted_optimizer.mariana_megatron import get_byted_optimizer_from_mariana_config

        optimizer_config_mariana_format = {
            "type": type,
            "params": {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "force_bfloat16_state": force_bfloat16_state,
            },
        }
        optimizer = get_byted_optimizer_from_mariana_config(param_groups, optimizer_config_mariana_format)
    except ImportError:
        warnings.warn(
            f"Cannot find {type} byted_optimizer. Try to use apex.optimizers instead. force_bfloat16_state is not supported."
        )
        try:
            from apex import optimizers

            if type in ["adam", "adamw"]:
                apex_optimizer = optimizers.FusedAdam
            elif type == "lamb":
                apex_optimizer = optimizers.FusedLAMB
            else:
                raise NotImplementedError
            optimizer = apex_optimizer(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        except ImportError:
            warnings.warn(
                f"Cannot find {type} in either byted_optimizer or apex.optimizers. Use torch.optim instead. force_bfloat16_state is not supported."
            )
            if type in ["adam", "adamw"]:
                import torch

                optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            else:
                raise NotImplementedError(f"Optimizer type: {type} is not supported.")
    return optimizer
