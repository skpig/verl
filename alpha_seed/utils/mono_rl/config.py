import copy
from omegaconf import DictConfig
from mono_rl.worker.actors.actor_config import (
    ActorWorkerConfig,
    CriticWorkerConfig,
    RefPolicyWorkerConfig,
    RewardWorkerConfig,
)

from omegaconf import OmegaConf
from mono_rl.worker.engine.fsdp.config import FSDPEngineConfig
from typing import Optional


def _set_config_field(mono_config, source_config: DictConfig, field: str):
    """ Set one specific field in a mono_rl dataclass config object"""
    if field in source_config and source_config[field] is not None:
        setattr(mono_config, field, source_config[field])


def _set_config_fsdp_engine(mono_config, source_config: DictConfig, model_config: Optional[DictConfig] = None):
    """ Set the fsdp_engine config of mono_config """

    assert isinstance(mono_config, FSDPEngineConfig), "mono_config must be a FSDPEngineConfig object"
    config = copy.deepcopy(source_config)

    # set top level config fields
    _set_config_field(mono_config, config, "ppo_micro_batch_size")
    _set_config_field(mono_config, config, "ppo_max_token_len")

    # set the hf_model_config field of the fsdp_engine
    _set_config_field(mono_config.model, config, "use_ce_loss_fusion")
    _set_config_field(mono_config.model, config, "logits_clamp")
    _set_config_field(mono_config.model, config, "update_gate_ema")
    _set_config_field(mono_config.model, config, "remove_o_bias")
    _set_config_field(mono_config.model, config, "freeze_gate")

    # set the hf_model_config field of the fsdp_engine from model_config
    source_model_config = model_config
    if not source_model_config and hasattr(config, "model"):
        source_model_config = config.model
    if source_model_config:
        _set_config_field(mono_config.model, source_model_config, "path")
        _set_config_field(mono_config.model, source_model_config, "external_lib")
        _set_config_field(mono_config.model, source_model_config, "use_rmpad")
        _set_config_field(mono_config.model, source_model_config, "enable_gradient_checkpointing")
        # special case for override_config as it is a dict
        if hasattr(source_model_config, "override_config"):
            override_config_clean_dict = OmegaConf.to_container(source_model_config.override_config, resolve=True)
            mono_config.model.override_config = override_config_clean_dict

    # set the hf_optim_config field of the fsdp_engine
    _set_config_field(mono_config.optim, config, "grad_clip")
    if hasattr(config, "optim"):
        _set_config_field(mono_config.optim, config.optim, "type")
        _set_config_field(mono_config.optim, config.optim, "lr")
        _set_config_field(mono_config.optim, config.optim, "betas")
        _set_config_field(mono_config.optim, config.optim, "eps")
        _set_config_field(mono_config.optim, config.optim, "weight_decay")
        _set_config_field(mono_config.optim, config.optim, "lr_warmup_steps")
        _set_config_field(mono_config.optim, config.optim, "lr_warmup_steps_ratio")
        _set_config_field(mono_config.optim, config.optim, "min_lr_ratio")
        _set_config_field(mono_config.optim, config.optim, "warmup_style")
        _set_config_field(mono_config.optim, config.optim, "total_training_steps")
        _set_config_field(mono_config.optim, config.optim, "force_bfloat16_state")

    # set the fsdp field of the fsdp_engine
    _set_config_field(mono_config.fsdp, config, "strategy")
    if mono_config.fsdp.strategy == "vescale-fsdp2":
        mono_config.fsdp.strategy = "vescale"
    if hasattr(config, "fsdp_config"):  # Note that param_offload is a sub-field of role.fsdp_config.param_offload
        mono_config.fsdp.param_offload = config.fsdp_config.param_offload
    _set_config_field(mono_config.fsdp, config, "fsdp_size")
    _set_config_field(mono_config.fsdp, config, "act_offload")
    _set_config_field(mono_config.fsdp, config, "act_offload_upbound")
    _set_config_field(mono_config.fsdp, config, "act_offload_buff_size")
    _set_config_field(mono_config.fsdp, config, "act_offload_threshold")
    _set_config_field(mono_config.fsdp, config, "ulysses_sequence_parallel_size")
    _set_config_field(mono_config.fsdp, config, "oe_size")
    _set_config_field(mono_config.fsdp, config, "tp_size")
    _set_config_field(mono_config.fsdp, config, "tp_outside")
    _set_config_field(mono_config.fsdp, config, "balance_tokens")

    # set the profile field of the fsdp_engine
    if hasattr(config, "profile"):
        _set_config_field(mono_config.profile, config.profile, "filename")
        _set_config_field(mono_config.profile, config.profile, "profile_on_ranks")
        _set_config_field(mono_config.profile, config.profile, "upload_to_mlx")
        _set_config_field(mono_config.profile, config.profile, "enable")
        _set_config_field(mono_config.profile, config.profile, "warmup")
        _set_config_field(mono_config.profile, config.profile, "wait")
        _set_config_field(mono_config.profile, config.profile, "active")
        _set_config_field(mono_config.profile, config.profile, "mem_enable")


def _set_config_megatron_engine(mono_config, source_config: DictConfig, model_config: Optional[DictConfig] = None):
    """ Set the megatron_engine config of mono_config """
    pass


def critic_config_to_mono_config(critic_config: DictConfig) -> CriticWorkerConfig:
    """ Convert a alpha-seed `DictConfig` to a mono_rl `CriticWorkerConfig`"""

    config = copy.deepcopy(critic_config)
    mono_config = CriticWorkerConfig()

    # set strategy according to the config
    _set_config_field(mono_config, config, "strategy")

    if mono_config.strategy in ["fsdp", "vescale-fsdp2"]:
        _set_config_fsdp_engine(mono_config.engine, config)
    elif mono_config.strategy == "megatron":
        _set_config_megatron_engine(mono_config.engine, config)
    else:
        raise NotImplementedError(f"Strategy {mono_config.strategy} is not supported")

    return mono_config


def reward_config_to_mono_config(reward_config: DictConfig) -> RewardWorkerConfig:
    """ Convert a alpha-seed `DictConfig` to a mono_rl `RewardWorkerConfig`"""

    config = copy.deepcopy(reward_config)
    mono_config = RewardWorkerConfig()

    # set strategy according to the config
    _set_config_field(mono_config, config, "strategy")

    if mono_config.strategy in ["fsdp", "vescale-fsdp2"]:
        _set_config_fsdp_engine(mono_config.engine, config)
    elif mono_config.strategy == "megatron":
        _set_config_megatron_engine(mono_config.engine, config)
    else:
        raise NotImplementedError(f"Strategy {mono_config.strategy} is not supported")

    return mono_config


def actor_config_to_mono_config(actor_config: DictConfig, model_config: DictConfig) -> ActorWorkerConfig:
    """ Convert a alpha-seed `DictConfig` to a mono_rl `ActorWorkerConfig`"""

    config = copy.deepcopy(actor_config)
    mono_config = ActorWorkerConfig()

    # set strategy according to the config
    _set_config_field(mono_config, config, "strategy")

    if mono_config.strategy in ["fsdp", "vescale-fsdp2"]:
        _set_config_fsdp_engine(mono_config.engine, config, model_config)
    elif mono_config.strategy == "megatron":
        _set_config_megatron_engine(mono_config.engine, config, model_config)
    else:
        raise NotImplementedError(f"Strategy {mono_config.strategy} is not supported")

    return mono_config


def ref_config_to_mono_config(ref_config: DictConfig, model_config: DictConfig) -> RefPolicyWorkerConfig:
    """ Convert a alpha-seed `DictConfig` to a mono_rl `RefPolicyWorkerConfig`"""

    config = copy.deepcopy(ref_config)
    mono_config = RefPolicyWorkerConfig()

    # set strategy according to the config
    _set_config_field(mono_config, config, "strategy")

    if mono_config.strategy in ["fsdp", "vescale-fsdp2"]:
        _set_config_fsdp_engine(mono_config.engine, config, model_config)
    elif mono_config.strategy == "megatron":
        _set_config_megatron_engine(mono_config.engine, config, model_config)
    else:
        raise NotImplementedError(f"Strategy {mono_config.strategy} is not supported")

    return mono_config
