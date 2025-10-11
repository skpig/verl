set -x
NUM_STEPS="${NUM_STEPS:-240}"
echo $NUM_STEPS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/m8_vlm_680m_seedvit
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/m8_vlm_680m_seedvit
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/data/rl/math_37k_knowlegde_species_15k_zero_train.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/eval_mathvision_mini.parquet
default_hdfs_dir=/opt/tiger/test/vlm_grpo_680m_async_mux_$(date +%F)
mkdir -p ${default_hdfs_dir}

# 训练长度
max_prompt_length=8192
max_response_length=1024
# batch size && 训练epoch

train_batch_size=8
ppo_mini_batch_size=8
val_batch_size=8
total_epochs=200
test_freq=-1
save_freq=-1
# 算法相关的参数
actor_lr=2e-6
critic_lr=2e-6
lr_warmup_steps=0
kl_coef=0.0
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
force_append_eos=True
upgo_loss_weight=0.0
upgo_loss_version=1
clip_ratio2=2.0
weight_decay=0.1
adv_estimator=gae
kl_loss_weight=0.0004
num_bon=1
bon_strategy=all
kl_penalty=low_var_kl

# tracking实验名
project_name='alpha_seed_vlm'
experiment_name="m8_2b5_grpo_$(date +%F)"

actor_sp_size=2
critic_sp_size=2
ref_sp_size=2
reward_sp_size=2
fsdp_size=2
xperf_tp_size=2

python3 tasks/main_ppo.py \
    algorithm.use_variable_lambda=True \
    algorithm.variable_lambda_scalar=0.05 \
    algorithm.use_separate_critic_lam=True \
    algorithm.critic_lam=1.0 \
    critic.ppo_epochs=1 \
    trainer.code_sandbox_psm='data.aml.code_sandbox_arnold_online.service.hl' \
    trainer.verifier_service_psm="'seed.alphaseed.verify_service?idc=yg&cluster=default'" \
    trainer.use_remote_sandbox=True \
    trainer.use_remote_verifier=True \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=img \
    data.dist_image=True \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.truncation='error' \
    +data.chat_template=seed \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.actor.scale_pg_by_kl=False \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.model.override_config.architectures=['SeedVLForTokenClassification'] \
    +critic.model.override_config.text_config.num_labels=1 \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    reward_model.last_response_sep=${last_response_sep} \
    +reward_model.use_rmpad=True \
    reward_model.reward_0_for_overlong_rsp=False \
    reward_model.punish_no_answer=v0 \
    reward_model.add_int_verify=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.critic_warmup=2 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    streaming_rollout.nnodes=1 \
    streaming_rollout.n_gpus_per_node=2 \
    streaming_validator.nnodes=1 \
    streaming_validator.n_gpus_per_node=2 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps="disable" \
    trainer.offload_train_memory=True \
    +actor_rollout_ref.rollout.complete_ratio=0.5 \
    +actor_rollout_ref.rollout.max_off_policy_steps=5 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    actor_rollout_ref.actor.shuffle=False \
    +actor_rollout_ref.rollout.use_vllm=True \
    actor_rollout_ref.rollout.enable_paged_attention=True \
    +actor_rollout_ref.rollout.dump_nan=${default_hdfs_dir}/dump_nan \
    streaming_rollout.warmup_step=0 \
    +actor_rollout_ref.rollout.use_ep=False \
    actor_rollout_ref.rollout.quant_mode=WFP8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.88 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${xperf_tp_size} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    critic.fsdp_size=${fsdp_size} \
    reward_model.fsdp_size=${fsdp_size} \
    +trainer.enable_actor_critic_spatial_mux=True \
    trainer.total_steps=${NUM_STEPS}