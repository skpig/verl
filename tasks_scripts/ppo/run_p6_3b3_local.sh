set -x
ray stop --force


# tracking log
model_size="3b3"  # in 400m, 3b3, 12B
project_name='alphaseed_scaling'
experiment_name=${model_size}'_dev_logit_manipulate'
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/shengdinghu/rl/${experiment_name}_${MERLIN_JOB_ID}

eval "$(python3 tasks_scripts/get_model_path.py --varname SFT_MODEL_PATH --size ${model_size} --type sft_eot_jiaze1126 --platform i18n_azure)"
echo "SFT_MODEL_PATH: "$SFT_MODEL_PATH

eval "$(python3 tasks_scripts/get_model_path.py --varname RM_MODEL_PATH --size ${model_size} --type prm --platform i18n_OCI)"
echo "RM_MODEL_PATH: "$RM_MODEL_PATH 

# SFT_MODEL_PATH="hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/models/3b3_jiaze_1126a1_eot/241114_3b3_sft30_12b-kd-bo128_hf"
# hdfs dfs -cp $SFT_MODEL_PATH $SFT_MODEL_PATH"_eottokv2"
# hdfs dfs -get hdfs://haruna/home/byte_data_seed/ssd_hldy/user/chenjiaze/alphaseed_workspace/bbpe155k-v6.4.3-ml.add_thinking_v2 outputs/tmp_tokenizer
# hdfs dfs -put -f outputs/tmp_tokenizer/* $SFT_MODEL_PATH"_eottokv2"
# hdfs dfs -ls $SFT_MODEL_PATH"_eottokv2"
# SFT_MODEL_PATH=$SFT_MODEL_PATH"_eottokv2"
# echo $SFT_MODEL_PATH
# exit 0

TRAIN_FILE=hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/training_data/released_data/release_1.3.parquet

TEST_FILE=[hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/AIME_evals.parquet] #,hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/CMO_evals.parquet,hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/IMO_evals.parquet,hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/MATH_evals.parquet]



# 训练长度
max_prompt_length=1024
max_response_length=2048
# batch size && 训练epoch
train_batch_size=8192
val_batch_size=696
ppo_mini_batch_size=1024
total_epochs=5000
test_freq=5
save_freq=5
# 算法相关的参数
actor_lr=1e-6
critic_lr=2e-6
lr_warmup_steps=10 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.003
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
force_append_eos=True
upgo_loss_weight=0.2
upgo_loss_version=1
# 工程参数
gen_micro_batch_size=512 # use_dynamic_bsz=True时仍然生效
infer_micro_batch_size=512 # use_dynamic_bsz=True时不生效
train_micro_batch_size=64 # use_dynamic_bsz=True时不生效
actor_sp_size=4
critic_sp_size=4
ref_sp_size=1
reward_sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=18432
critic_ppo_max_token_len=18432
infer_ppo_max_token_len=36864
fsdp_size=-1
gen_tp=4
offload=True


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
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    +actor_rollout_ref.rollout.summary_min_space=1024 \
    +actor_rollout_ref.rollout.soft_interval=512 \
    +actor_rollout_ref.rollout.enable_eot=False \
    actor_rollout_ref.rollout.eval_bon=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.scale_pg_by_kl=True \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_max_token_len=${critic_ppo_max_token_len} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.infer_micro_batch_size=${infer_micro_batch_size} \
    critic.model.fsdp_config.param_offload=${offload} \
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
    reward_model.reward_0_for_overlong_rsp=False \
    reward_model.punish_no_answer=v0 \
    reward_model.use_dynamic_bsz=${use_dynamic_bsz} \
    reward_model.max_token_len=${infer_ppo_max_token_len} \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${ARNOLD_WORKER_NUM} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=True \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps=auto