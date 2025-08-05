set -x

NUM_STEPS="${NUM_STEPS:-1}"
echo $NUM_STEPS

export THINK_TEMPLATE=v1

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/checkpoints/xperf/m8_2b5_32k_seedvit_400m_baseline_openthought_8k_simplified_sys_fix_dropout_rope50

TRAIN_FILE=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/xiaoboqin/data/rlhf/math/mmathcot_v4_hard_w_sys_for_rl.parquet

# init from policy
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/m8_2b5_ppo_8k_xperf_0509/global_step_40/critic/huggingface/huggingface

TEST_FILE=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/data/rl/eval_alphaseed_mathvision_fix_v2_dot.parquet
project_name='alphaseed_nightly_ci'
experiment_name="regression_vlm_m8_2b5_ppo_$(date +%F)"

default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/${project_name}/${experiment_name}
save_train_batch_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/${project_name}/${experiment_name}/batch_data
echo "default_hdfs_dir ${default_hdfs_dir}"

# 训练长度
max_prompt_length=8192
max_response_length=4096
# batch size && 训练epoch
train_batch_size=4096
val_batch_size=3039
ppo_mini_batch_size=256
critic_warmup=0

total_epochs=100
test_freq=5
save_freq=-1
# 算法相关的参数
actor_lr=1e-6
critic_lr=2e-6
loss_average_method=token
lr_warmup_steps=10 #/ (train_size * total_epochs / train_batch_size)
kl_coef=1e-5
use_last_response=False
use_ref_answer=False
gae_gamma=1.0
gae_lam=0.95
force_append_eos=True
upgo_loss_weight=0
upgo_loss_version=1
clip_ratio2=2.0
weight_decay=0.1
kl_loss_weight=0.0
kl_penalty=low_var_kl
temperature=1.0
use_separate_critic_lam=True
critic_lam=1.0

# 工程参数
gen_micro_batch_size=256
infer_micro_batch_size=512
train_micro_batch_size=64

actor_ppo_max_token_len=155296
critic_ppo_max_token_len=155296
actor_ref_max_token_len=200000

fsdp_size=-1

cd alpha-seed

python3 tasks/main_ppo.py \
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
    data.val_batch_size=${val_batch_size} \
    data.truncation='left' \
    +data.chat_template=seed \
    actor_rollout_ref.rollout.train_generate_kwargs.temperature=${temperature} \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    +actor_rollout_ref.model.override_config.text_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.text_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.text_config.resid_pdrop=0. \
    +actor_rollout_ref.model.override_config.vision_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.vision_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.vision_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.loss_average_method=${loss_average_method} \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.infer_micro_batch_size=${infer_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.model.override_config.text_config.attention_dropout=0. \
    +critic.model.override_config.text_config.embd_pdrop=0. \
    +critic.model.override_config.text_config.resid_pdrop=0. \
    +critic.model.override_config.vision_config.attention_dropout=0. \
    +critic.model.override_config.vision_config.embd_pdrop=0. \
    +critic.model.override_config.vision_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    critic.fsdp_size=${fsdp_size} \
    reward_model.fsdp_size=${fsdp_size} \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=${infer_micro_batch_size} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=True \
    reward_model.need_punish_trunc=False \
    reward_model.trunc_punish_score=-0.1 \
    reward_model.punish_format=True \
    reward_model.format_punish_score=-0.1 \
    +reward_model.log_image=False \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    algorithm.kl_penalty=${kl_penalty} \
    algorithm.mask_overlong=False \
    algorithm.use_separate_critic_lam=${use_separate_critic_lam} \
    algorithm.critic_lam=${critic_lam} \
    trainer.eval_before_training=False \
    trainer.critic_warmup=${critic_warmup} \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_train_batch_dir=${save_train_batch_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps="auto" \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    +actor_rollout_ref.rollout.max_ctx_batch_size=1 \
    actor_rollout_ref.rollout.enable_paged_attention=True \
    trainer.save_cases_to_hdfs=False \
    trainer.total_steps=${NUM_STEPS} \
    actor_rollout_ref.actor.ppo_max_token_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.max_token_len=${actor_ref_max_token_len} \
    critic.ppo_max_token_len=${critic_ppo_max_token_len} \
    trainer.offload_train_memory=True \
    actor_rollout_ref.model.use_ce_loss_fusion=True