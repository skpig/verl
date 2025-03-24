set -x
ray stop --force


# tracking log
model_size="400m"  # in 400m, 3b3, 12B
project_name='alphaseed_scaling'
experiment_name=${model_size}'_dev_logit_manipulate'
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/shengdinghu/rl/${experiment_name}_${MERLIN_JOB_ID}

eval "$(python3 tasks_scripts/get_model_path.py --varname SFT_MODEL_PATH --size ${model_size} --type sft_baseline --platform i18n_OCI)"
echo "SFT_MODEL_PATH: "$SFT_MODEL_PATH

eval "$(python3 tasks_scripts/get_model_path.py --varname RM_MODEL_PATH --size ${model_size} --type rm_baseline --platform i18n_OCI)"
echo "RM_MODEL_PATH: "$RM_MODEL_PATH 

# SFT_MODEL_PATH=hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_sft/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf
# hdfs dfs -cp $SFT_MODEL_PATH $SFT_MODEL_PATH"_eottokv2"
# hdfs dfs -get hdfs://haruna/home/byte_data_seed/ssd_hldy/user/chenjiaze/alphaseed_workspace/bbpe155k-v6.4.3-ml.add_thinking_v2 outputs/tmp_tokenizer
# hdfs dfs -put -f outputs/tmp_tokenizer/* $SFT_MODEL_PATH"_eottokv2"
# hdfs dfs -ls $SFT_MODEL_PATH"_eottokv2"
# SFT_MODEL_PATH=$SFT_MODEL_PATH"_eottokv2"

# exit 0

TRAIN_FILE=hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/training_data/released_data/release_1.3.parquet

TEST_FILE=[hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/AIME_evals.parquet,hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/CMO_evals.parquet,hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/IMO_evals.parquet,hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/MATH_evals.parquet]



# 训练长度
max_prompt_length=2048
max_response_length=2048
# batch size && 训练epoch
train_batch_size=1024
val_batch_size=500
ppo_mini_batch_size=128
total_epochs=5000
test_freq=5
save_freq=10
# 算法相关的参数
actor_lr=2e-6
critic_lr=2e-6
lr_warmup_steps=1 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.0001
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
force_append_eos=True
upgo_loss_weight=0.2
upgo_loss_version=1
clip_ratio2=2.0
# 工程参数
gen_micro_batch_size=512
infer_micro_batch_size=512
train_micro_batch_size=64
ulysses_sequence_parallel_size=1
offload_train_mem=True
fsdp_size=2


python3 tasks/main_ppo.py \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    trainer.resume_steps=disable \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.truncation='left' \
    +data.chat_template=seed \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.complete_ratio=0.5 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=True \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    +actor_rollout_ref.rollout.summary_min_space=1024 \
    +actor_rollout_ref.rollout.soft_interval=512 \
    +actor_rollout_ref.rollout.enable_eot=True \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.actor.scale_pg_by_kl=True \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.infer_micro_batch_size=${infer_micro_batch_size} \
    critic.fsdp_size=${fsdp_size} \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.fsdp_size=${fsdp_size} \
    reward_model.micro_batch_size=${infer_micro_batch_size} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=True \
    reward_model.reward_0_for_overlong_rsp=False \
    reward_model.punish_no_answer=v0 \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.offload_train_memory=${offload_train_mem} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl\
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    streaming_rollout.nnodes=1 \
    streaming_rollout.n_gpus_per_node=4 \
    streaming_rollout.warmup_step=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    streaming_rollout.force_eos=True