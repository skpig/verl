from seed_models import M8Config
from mariana.models.text.config import TrainConfig


def convert_hf_config_to_mariana(hf_config: M8Config, model_implementation):
    #TODO(zhangchi.usc1992):
    assert isinstance(hf_config, M8Config)

    noop_transformer_layers = model_implementation.noop_transformer_layers
    if noop_transformer_layers is None:
        noop_transformer_layers = []

    # harcode for m8 for now
    layer_norm_type = 'rmsnorm'
    scale_attn_by_inverse_layer_idx = False
    tie_weight = True
    position_embeddings_type = "rope"

    n_shared_qhead = hf_config.num_attention_heads // hf_config.num_key_value_heads

    # note that in mariana, kv_mirror_layers starts from 1. It doens't include no-op layer when indexing
    if hf_config.kv_mirror_layers is not None:
        kv_mirror_layers = [i + 1 for i in hf_config.kv_mirror_layers]
        kv_mirror_imitated_layers = [i + 1 for i in hf_config.kv_mirror_imitated_layers]
    else:
        kv_mirror_layers = None
        kv_mirror_imitated_layers = None

    if hf_config.pre_post_layernorm_layers is not None:
        residual_post_ln_layers = [i + 1 for i in hf_config.pre_post_layernorm_layers]
    else:
        residual_post_ln_layers = None

    model_config = TrainConfig(
        hidden_size=hf_config.hidden_size,
        n_embed=hf_config.hidden_size,  # vocab embedding
        n_inner=hf_config.intermediate_size,
        n_head=hf_config.num_attention_heads,
        n_layer=hf_config.num_hidden_layers + len(noop_transformer_layers),
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        layer_norm_epsilon=hf_config.rms_norm_eps,
        activation_function="gelu_new", # this is useless
        #
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,  # TODO:
        initializer_range=0.02,  # this doesn't matter for post-training
        tie_weight=tie_weight,
        pad_idx=1,
        position_embeddings_type=position_embeddings_type,
        n_shared_qhead=n_shared_qhead,
        num_q_heads=-1,
        num_kv_heads=-1,
        head_dim=-1,
        kv_mirror_layers=kv_mirror_layers,  # layers which use kv cache from other layers, such as [8,9,10,11]
        kv_mirror_imitated_layers=kv_mirror_imitated_layers,  # layers which provide kv cache for other layers, such as [1,2,3,4]
        hidden_decoding_layers=None,  # layers which use other layers weight, such as [5,6,7,8]
        hidden_decoding_imitated_layers=None,  # layers weight used by other layers, such as [1,2,3,4]
        residual_post_ln_layers=residual_post_ln_layers,  # layers which residual using layernorm output, such as [1,2,3,4]
        repeat_kv_heads=False,
        sparse_attention_window_size=None,  # TODO: add sparse attention
        use_query_swiglu=False,
        query_swiglu_inner_dim=8192,
        dense_ffn_layers=None,
        dense_ffn_type='default',
        dense_ffn_inner_dim=-1,
        moe_expert_type="swiglu-melego-ep",
        moe_gate_type="cap-lego-ep",
        moe_gate_metric_type="lego",
        moe_expert_exp_level=4,
        moe_expert_exp_first_dim_factor=1.0,
        moe_expert_exp_first_num=2,
        moe_topk=hf_config.moe_topk,
        moe_num_expert=hf_config.moe_num_expert,
        moe_expert_eq_dim_factor=1,
        moe_backend="janus",  # options are: "janus", "default"
        moe_aux_loss_weight=0.00, # not used in rl
        moe_gate_dropout=0.0, # not used in rl
        moe_use_balance=False,   # in rl, this is always false
        moe_expert_group_capacity=1.0,  # for balance, not used in rl
        moe_expert_group_balance_loss_weight=0.0,  # for balance, not used in rl
        moe_expert_groups_in_ep_rank=1,  # for balance, not used in rl
        moe_enable_warmup=False,  # for balance, not used in rl
        moe_swiglu_fc1_2_init_scale=1.0,  # not used in rl
        convert_gate_to_fp32=False,
        moe_enable_ema_update=-1,  # -1 means lora training = 0, full training = 1
        query_head_scale_factor=2,
        # shared experts
        moe_pr_scale_factor=hf_config.share_expert_num,
        moe_pr_expert_type=model_implementation.moe_pr_expert_type,  # whether the shared expert is chunked by tp  'swiglu-default-duplicate'
        lora_rank=0,  # qkv_lora for continue train, legacy config, checkout qkv_lora_ranks
        rope_mode=hf_config.rope_scaling['rope_type'],  # rope mode options are: "default" "ntk" "scale"
        rope_base=hf_config.rope_theta,  # rope base.
        rope_scale=hf_config.rope_scaling['factor'],
        rope_cut=hf_config.rope_scaling['rope_cut'],
        rope_cut_head_dim=hf_config.rope_scaling['rope_cut_head_dim'],
        rope_force_fp32=False,
        sparse_attention_window_scale=1,
        sparse_attention_global_window_size=None,
        use_attention_bias=hf_config.attention_bias,
        layer_norm_type=layer_norm_type,
        exact_token_as_loss_denominator=False,  # this only affect loss in pretrain/sft
        use_key_layernorm=hf_config.use_key_layernorm,
        key_norm_after_rope=False,
        use_query_layernorm=False,
        use_context_groupnorm=hf_config.use_context_groupnorm,
        use_mariana_gqa_pattern=False,
        hyperconnection_rate=-1,
        deterministic_mode=False,
        # ======== configurations for efficiency ========
        cross_entropy_spilt_num=1,
        use_xperf_rotary=False,
        fuse_gelu_gemm=True,
        force_mem_efficient_layers=None,
        noop_transformer_layers=noop_transformer_layers,
        moe_overlap_recomp_grad_comm=False,
        moe_expert_op_version='V6',  # not used
        janus_use_big_op=model_implementation.janus_use_big_op,
        janus_big_op_version=model_implementation.janus_big_op_version,
        janus_big_op_attn_grad_accum_fusion=True,
        janus_p7_big_op_mlp_fwd_rs_fp8_compression="",
        janus_p7_big_op_mlp_bwd_ag_fp8_compression="",
        use_sequence_parallel_attention=False,  # sp attn
        use_sequence_parallel_attention_a2a=False,
        context_parallel_use_all_gather = False,
        enable_hybrid_data_parallel = False,
        cross_entropy_fusion = 'none',
        rope_gen_method = 'loader',
        fp8_use_bf16_layers='',
        # ========== configurations for training algos ========
        skip_n_iters=-1,
        save_mixed_ckpt_in_shards=False,
        save_mixed_model_states_freq='final',
        cont_train_mode="default",  # cont_train mode options are: "default" "lora"
        fuse_lora_weight=True,
    )

    model_config.set_params_dtype('bf16')
    return model_config


def update_megatron_config(model_config, megatron_config, vpp_size):
    vpp_layers_per_stage = None
    if vpp_size > 1 and megatron_config.pipeline_parallel_size > 1:
        pp_size = megatron_config.pipeline_parallel_size
        vpp_layers_per_stage = model_config.n_layer // vpp_size // pp_size
        assert model_config.n_layer == vpp_layers_per_stage * pp_size * \
            vpp_size, f"{model_config.n_layer} != {vpp_layers_per_stage} * {pp_size} * {vpp_size}"
    megatron_config.num_layers_per_virtual_pipeline_stage = vpp_layers_per_stage
    megatron_config.update_parallel_cfg()
