set -x

ray stop --force

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_STEPS="${NUM_STEPS:-2000}"
echo $NUM_STEPS

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/m10_680m_new
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_moe_400m_0716_sftv27_stage2_hf
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/test_with_ref_ans.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/test/m10_680m_grpo

# 训练长度
max_prompt_length=2048
max_response_length=2048
# batch size && 训练epoch
train_batch_size=128
ppo_mini_batch_size=256
val_batch_size=5000
total_epochs=100
test_freq=5
save_freq=-1
# 算法相关的参数
actor_lr=1e-6
critic_lr=2e-6
lr_warmup_steps=10
kl_coef=0.00
use_last_response=false
use_ref_answer=true
gae_gamma=1.0
gae_lam=0.95
force_append_eos=true
upgo_loss_weight=0.0
upgo_loss_version=1
clip_ratio2=2.0
weight_decay=0.1
adv_estimator=grpo
kl_loss_weight=0.00
num_bon=16
bon_strategy=all
kl_penalty=low_var_kl
# tracking实验名
project_name='verl_example_math_ci'
experiment_name='m10-680m-vescale'

ppo_max_token_len_per_gpu=32768
ppo_infer_max_token_len_per_gpu=32768
actor_sp_size=4
critic_sp_size=4
ref_sp_size=4
actor_tp_size=4
act_offload=true

actor_ppo_max_token_len=$((ppo_max_token_len_per_gpu * actor_sp_size))
critic_ppo_max_token_len=$((ppo_max_token_len_per_gpu * critic_sp_size))
infer_ppo_max_token_len=$((ppo_infer_max_token_len_per_gpu * ref_sp_size))

fsdp_size=-1
xperf_tp_size=2
strategy='vescale-fsdp2'

python3 tasks/main_ppo.py \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.truncation='left' \
    +data.chat_template=seed \
    actor_rollout_ref.rollout.name=xperf_gpt \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${xperf_tp_size} \
    actor_rollout_ref.rollout.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    +actor_rollout_ref.rollout.use_vllm=true \
    +actor_rollout_ref.rollout.use_ep=true \
    +actor_rollout_ref.rollout.vocab_tp=true \
    +actor_rollout_ref.rollout.use_mtp=true \
    +actor_rollout_ref.rollout.complete_ratio=1.0 \
    +actor_rollout_ref.rollout.max_off_policy_steps=5 \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.actor.strategy=${strategy} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.actor.scale_pg_by_kl=false \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.ppo_max_token_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.tp_size=${actor_tp_size} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.actor.act_offload=${act_offload} \
    actor_rollout_ref.actor.balance_tokens=true \
    actor_rollout_ref.ref.strategy=${strategy} \
    actor_rollout_ref.ref.tp_size=${actor_tp_size} \
    actor_rollout_ref.ref.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.ref.balance_tokens=true \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=true \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    critic.model.external_lib=seed_models \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=true \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=true \
    critic.strategy=${strategy} \
    critic.ppo_max_token_len=${critic_ppo_max_token_len} \
    critic.fsdp_size=${fsdp_size} \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    critic.tp_size=${actor_tp_size} \
    critic.balance_tokens=true \
    reward_model.enable=false \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=true \
    reward_model.reward_0_for_overlong_rsp=false \
    reward_model.punish_no_answer=v0 \
    reward_model.max_token_len=${infer_ppo_max_token_len} \
    reward_model.add_int_verify=false \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=true \
    trainer.val_only=false \
    trainer.val_epoch=1 \
    trainer.need_log=false \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps="disable" \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    trainer.offload_train_memory=true \
    trainer.total_steps=${NUM_STEPS} \
    2>&1 | tee log.txt
