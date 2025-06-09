set -x
NUM_STEPS="${NUM_STEPS:-240}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
echo $NUM_STEPS

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/m8_vlm_680m_seedvit

TRAIN_FILE=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/xiaoboqin/data/rlhf/math/mmathcot_v4_hard_w_sys_for_rl.parquet
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/data/rl/math_37k_knowlegde_species_15k_zero_train.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/eval_mathvision_mini.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/test/vlm_grpo


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
kl_coef=0.00001
use_last_response=False
use_ref_answer=False
gae_gamma=1.0
gae_lam=0.95
force_append_eos=True
upgo_loss_weight=0.0
upgo_loss_version=1
clip_ratio2=2.0
weight_decay=0.1
adv_estimator=grpo
kl_loss_weight=0.0004
num_bon=2
bon_strategy=all
kl_penalty=low_var_kl
temperature=1.2

# tracking实验名
project_name='alpha_seed_vlm'
experiment_name="m8_2b5_grpo_$(date +%F)"
# 工程参数
gen_micro_batch_size=8 # use_dynamic_bsz=True时仍然生效
infer_micro_batch_size=8 # use_dynamic_bsz=True时不生效
train_micro_batch_size=8 # use_dynamic_bsz=True时不生效

use_dynamic_bsz=True
actor_ppo_max_token_len=36864
critic_ppo_max_token_len=36864
infer_ppo_max_token_len=36864

actor_sp_size=1
critic_sp_size=1
ref_sp_size=1
reward_sp_size=1
fsdp_size=-1
xperf_tp_size=4
offload=True
offload_train_memory=True

python3 tasks/main_ppo.py \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=img \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.truncation='error' \
    +data.chat_template=seed \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.train_generate_kwargs.temperature=${temperature} \
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
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.scale_pg_by_kl=False \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps="disable" \
    +actor_rollout_ref.rollout.complete_ratio=1.0 \
    +actor_rollout_ref.rollout.max_off_policy_steps=0 \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    reward_model.need_punish_trunc=True \
    reward_model.trunc_punish_score=-0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${xperf_tp_size} \
    +actor_rollout_ref.rollout.use_vllm=False \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    trainer.offload_train_memory=${offload_train_memory} \
    critic.profile.enable=False \
    critic.profile.upload_to_mlx=False \
    critic.profile.filename=actor.tp${xperf_tp_size}.fsdp${fsdp_size} \
    actor_rollout_ref.actor.profile.enable=False \
    actor_rollout_ref.actor.profile.upload_to_mlx=False \
    actor_rollout_ref.actor.profile.filename=actor.tp${xperf_tp_size}.fsdp${fsdp_size} \
    trainer.total_steps=${NUM_STEPS} \
    trainer.save_cases_to_hdfs=False
