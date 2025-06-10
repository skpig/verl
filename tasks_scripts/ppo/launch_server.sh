set -x

export SEC_KV_AUTH=1
export BYTED_RAY_DISABLE_COLOR_LOG=true
export HDFS_IO_THROW_EXCEPTION=1
export USE_FLASH_ATTENTION_2=1
export PYTHONPATH=$PYTHONPATH:/opt/tiger/alpha-seed:/opt/tiger/mono_rl:/opt/tiger/verl:/opt/tiger/verifiable_tasks:/opt/tiger/bpex_triton:/opt/tiger/seed_models

check_env_var_default() {
  local var_name=$1
  local default_val=$2
  if [[ -z "${!var_name}" ]]; then
    echo "WARN: Environment variable '$var_name' is not set. use default ${default_val}" >&2
    echo "${default_val}"
  else
    echo "${!var_name}"
  fi
}

check_env_var() {
  local var_name=$1
  if [[ -z "${!var_name}" ]]; then
    echo "ERROR: Environment variable '$var_name' is not set." >&2
    exit 1
  else
    echo "${!var_name}"
  fi
}

######### Set Default Env Var ##########
max_prompt_length=$(check_env_var_default "MAX_PROMPT_LENGTH" "2048")
max_response_length=$(check_env_var_default "MAX_RESPONSE_LENGTH" "16384")
gen_mp=$(check_env_var_default "GEN_MP" "auto")

######### Env Var to Variables #########
policy_model=$(check_env_var "POLICY_MODEL")
value_model=$(check_env_var "VALUE_MODEL")

role=${ARNOLD_ROLE^^}
role_gpu_per_node_var_name="ARNOLD_${role}_GPU"
role_nnode_var_name="ARNOLD_${role}_NUM"

role_gpu_per_node=${!role_gpu_per_node_var_name:-8}
role_nnode=${!role_nnode_var_name:-1}

python3 tasks/main_ppo.py \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    critic.model.path=${value_model} \
    reward_model.model.path=${value_model} \
    actor_rollout_ref.model.path=${policy_model} \
    actor_rollout_ref.model.external_lib=seed_models \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_mp} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=True \
    +actor_rollout_ref.rollout.enable_paged_attention=True \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=${role_gpu_per_node} \
    trainer.nnodes=${role_nnode} \
    server_client.role=server
