set -x

NUM_STEPS="${NUM_STEPS:-240}"
echo $NUM_STEPS

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_moe_400m_0716_sftv27_stage2_hf
TRAIN_FILE0=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TRAIN_FILE1=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TRAIN_FILE="[$TRAIN_FILE0,$TRAIN_FILE1]"
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/test_with_ref_ans.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/test/p6_400m_omnistore_test_1

# 训练长度
max_prompt_length=1024 # 16384
max_response_length=2048 # 16384
# batch size && 训练epoch
train_batch_size=1024
val_batch_size=5000
ppo_mini_batch_size=128
ppo_micro_batch_size=64
total_epochs=100
test_freq=10
save_freq=-1
# 算法相关的参数
actor_lr=1e-6
critic_lr=1e-5
lr_warmup_steps_ratio=0.0003 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.001
entropy_coeff=0.0001
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
kl_penalty=low_var_kl
# 工程参数
actor_sp_size=2
critic_sp_size=2
ref_sp_size=1
reward_sp_size=1

# tracking实验名
project_name='verl_example_math_ci'
experiment_name='p6_400m_math-v1'
export PYTHONPATH=$PYTHONPATH:/opt/tiger/verl:/opt/tiger/seed_models:/opt/tiger/verifiable_tasks:/opt/tiger/bpex_triton
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
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.rollout.micro_batch_size=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=512 \
    actor_rollout_ref.ref.ema=0.99 \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${ppo_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=True \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=512 \
    reward_model.mean=0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    reward_model.reward_0_for_overlong_rsp=True \
    reward_model.punish_no_answer=v0 \
    +reward_model.use_rmpad=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=True \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps=disable \
    trainer.set_fake_attention_mask=False \
    trainer.fake_seqlen_ratio=0.5 \
    streaming_rollout.nnodes=0 \
    streaming_rollout.n_gpus_per_node=4 \
    streaming_rollout.warmup_step=0 \
    streaming_rollout.force_eos=True \
    trainer.total_steps=${NUM_STEPS}