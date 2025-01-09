set -x

ray stop --force

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/jiangchengquan/alphaseed/2b525bmoe_P7_data1205ds_trial1207a1v2_hf
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_moe_400m_0716_sftv27_stage2_hf


# 训练长度
max_prompt_length=10240
max_response_length=22528

# tracking实验名
export PYTHONPATH=$PYTHONPATH:/opt/tiger/verl:/opt/tiger/seed_models:/opt/tiger/verifiable_tasks:/opt/tiger/bpex_triton

trap "kill 0" EXIT

exec python3 tasks/main_ppo.py \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    critic.model.path=${RM_MODEL_PATH} \
    reward_model.model.path=${RM_MODEL_PATH} \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.model.external_lib=seed_models \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.enable_paged_attention=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    server_client.role=server &

INPUT_FILES="['hdfs://haruna/home/byte_data_seed/hdd_hldy/user/wangchengyi/data/infer_input/alphaseed/AIME_repeat128_w_sp.parquet']"
OUTPUT_FILE="./gen_output.parquet"
TOKENIZER_PATH="hdfs://haruna/home/byte_data_seed/hdd_hldy/user/binxingyan/bbpe155k-v6.4.3-ml.pret"

python3 tasks/gen_client/main_gen_client.py \
        ray.server_addr=auto \
        data.tokenizer=${TOKENIZER_PATH} \
        data.input_files=${INPUT_FILES} \
        data.output_file=${OUTPUT_FILE} \
        data.prompt_key='problem' \
        data.max_prompt_length=${max_prompt_length} \
        gen.batch_size=8 \
        eval.sample_num=1 \
        eval.bon_list=[1]
