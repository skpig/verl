from cruise.utilities.cloud_io import load as crs_load
from datetime import datetime

from torch import distributed as dist

from mariana.utils.checkpoint_utils import (load_state_dict_in_shards_to_rank, load_state_dict_to_megatron,
                                            fake_quant_weight)


def rank_zero_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def load_partial_pretrain(gpt, partial_pretrain, model_config, lora_config=None, download_in_shards=True):
    """
    This function should act like AutoModelForCausalLM.from_pretrained,
    """

    def normalize_state_dict(state_dict):
        if 'module' in state_dict:
            state_dict = state_dict['module']
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        return state_dict

    # load megatron ckpt
    rank_zero_print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} start to load checkpoint '
                    f'to a megatron partitioned model')
    if download_in_shards:
        state_dict = load_state_dict_in_shards_to_rank(partial_pretrain, rank=0, map_location='cpu')
        if dist.get_rank() == 0:
            state_dict = normalize_state_dict(state_dict)
    else:
        if dist.get_rank() == 0:
            state_dict = crs_load(partial_pretrain, map_location='cpu')
            state_dict = normalize_state_dict(state_dict)
        else:
            state_dict = None

    # if (
    #     state_dict is not None
    #     and 'client_state' in state_dict
    #     and 'training trajectory' in state_dict['client_state']
    # ):
    #     self.training_trajectory = state_dict['client_state']['training trajectory']

    if state_dict is not None and lora_config is not None and lora_config.default.use_qlora:
        rank_zero_print('[QLoRA Log] quant pretrained weight to ', lora_config.default.qlora_quant_weight_dtype)
        fake_quant_weight(
            state_dict,
            lora_config,
            model_config.use_mariana_gqa_pattern,
            model_config.query_head_scale_factor,
        )
    dist.barrier()

    rank_zero_print(
        f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} loading checkpoint to a megatron partitioned model')

    moe_ckpt_version = 'M8'  # hardcode for now

    # TODO(zhangchi.usc1992): move this back once
    # use_qlora=lora_config.default.use_qlora,
    # quant_real_store=lora_config.default.qlora_quant_real_store,
    # quant_aware_L4Q=lora_config.default.qlora_quant_aware_L4Q,

    load_state_dict_to_megatron(
        state_dict,
        gpt,
        noop_layers=model_config.noop_transformer_layers,
        gate_fp32=model_config.convert_gate_to_fp32,
        use_mariana_gqa_pattern=model_config.use_mariana_gqa_pattern,
        query_head_scale_factor=model_config.query_head_scale_factor,
        moe_ckpt_version=moe_ckpt_version,
        tie_weight=model_config.tie_weight,
    )
