set -x

# tracking log
model_size="400m"  # in 400m, 3b3, 12B
project_name='alphaseed_scaling'
experiment_name=${model_size}'_dev'
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/shengdinghu/rl/${experiment_name}_${MERLIN_JOB_ID}

eval "$(python3 tasks_scripts/get_model_path.py --varname SFT_MODEL_PATH --size ${model_size} --type sft_baseline --platform i18n_OCI)"
echo "SFT_MODEL_PATH: "$SFT_MODEL_PATH

eval "$(python3 tasks_scripts/get_model_path.py --varname RM_MODEL_PATH --size ${model_size} --type rm_baseline --platform i18n_OCI)"
echo "RM_MODEL_PATH: "$RM_MODEL_PATH 

# data
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/hard60_format_repeat10.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/math_500.parquet


# ----- 算法相关的参数 ------
# 训练长度
max_response_length=8192 # 16384
# batch size
train_batch_size=1024
ppo_mini_batch_size=128
# lr
actor_lr=1e-6
critic_lr=2e-6
lr_warmup_steps=1 # 10 / (train_size * total_epochs / train_batch_size)
# 系数
kl_coef=0.0001
gae_gamma=1.0
gae_lam=0.95
upgo_loss_weight=0.2
upgo_loss_version=1
clip_ratio2=2.0
actor_entropy_coeff=0.0
# 其他
force_append_eos=True
reward_0_for_overlong_rsp=False
# sampling
num_bon=1
bon_strategy=best_worst

# ------- 固定参数 -------
max_prompt_length=2048
val_batch_size=500
total_epochs=500
test_freq=5
save_freq=50
use_last_response=False
use_ref_answer=True
eval_bon=1
eval_bon_every=20
use_dynamic_bsz=True

# ------- 不同规模各自的工程参数 ------
if [ "${model_size}" = "400m" ]; then
    gen_micro_batch_size=512
    infer_micro_batch_size=512
    train_micro_batch_size=64
    actor_ppo_max_token_len=36864
    critic_ppo_max_token_len=36864
    infer_ppo_max_token_len=73728
    rollout_tensor_model_parallel_size=1
    actor_sp_size=2
    critic_sp_size=2
    ref_sp_size=1
    reward_sp_size=1
    nnodes=4
    streaming_rollout_nnodes=1
elif [ "${model_size}" = "3b3" ]; then
    gen_micro_batch_size=1024
    infer_micro_batch_size=512
    train_micro_batch_size=64
    actor_ppo_max_token_len=36864
    critic_ppo_max_token_len=36864
    infer_ppo_max_token_len=73728
    rollout_tensor_model_parallel_size=4
    actor_sp_size=2
    critic_sp_size=2
    ref_sp_size=1
    reward_sp_size=1
    nnodes=4
    streaming_rollout_nnodes=1
elif [ "${model_size}" = "12B" ]; then
    gen_micro_batch_size=256
    infer_micro_batch_size=512
    train_micro_batch_size=32
    actor_ppo_max_token_len=36864
    critic_ppo_max_token_len=36864
    infer_ppo_max_token_len=73728
    rollout_tensor_model_parallel_size=4
    actor_sp_size=2
    critic_sp_size=2
    ref_sp_size=1
    reward_sp_size=1
    nnodes=4
    streaming_rollout_nnodes=1
else
    echo "Model size is not recognized, taking default actions."
    exit 1
fi


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
    data.multi_prompts=all \
    data.num_prompts_per_data=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=${actor_entropy_coeff} \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.eval_bon=${eval_bon} \
    actor_rollout_ref.rollout.eval_bon_every=${eval_bon_every} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.actor.scale_pg_by_kl=True \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_max_token_len=${critic_ppo_max_token_len} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.infer_micro_batch_size=${infer_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=${infer_micro_batch_size} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=True \
    reward_model.reward_0_for_overlong_rsp=${reward_0_for_overlong_rsp} \
    reward_model.punish_no_answer=v0 \
    reward_model.use_dynamic_bsz=${use_dynamic_bsz} \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    reward_model.max_token_len=${infer_ppo_max_token_len} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=True \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.save_cases_to_hdfs=True \
    +actor_rollout_ref.rollout.complete_ratio=0.8 \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=8 \
    streaming_rollout.nnodes=${streaming_rollout_nnodes} \
    streaming_rollout.n_gpus_per_node=8 \
    trainer.resume_steps=auto
