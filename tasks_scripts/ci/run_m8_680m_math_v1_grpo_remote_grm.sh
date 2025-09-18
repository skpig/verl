set -x
ray stop --force

export NCCL_DEBUG=WARN
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

NUM_STEPS="${NUM_STEPS:-120}"
echo $NUM_STEPS

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_moe_400m_0716_sftv27_stage2_hf
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/test_with_ref_ans_top_100.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/test/p6_400m_omnistore_test_1

use_grm_reverse=False
rm_psm="data.aml.arnold_inference_57781906"
rm_idc="'lq,lf,hl,yg,gl,wlby'"
rm_cluster="default"
rm_model_name="vlm/M8-23B-MoE-0820-mixrl_grm_merge_s180_xya_debug"

# 训练长度
max_prompt_length=128
max_response_length=128
# batch size && 训练epoch
train_batch_size=32
ppo_mini_batch_size=32
val_batch_size=5000
total_epochs=100
test_freq=5
save_freq=-1
# 算法相关的参数
optimizer_type=${OPTIM-adam}
force_bfloat16_state=${BF16STATE-False}
actor_lr=1e-5
critic_lr=2e-6
lr_warmup_steps=10
kl_coef=0.0
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
kl_loss_weight=0.1
num_bon=1
bon_strategy=all
kl_penalty=low_var_kl
# tracking实验名
project_name='verl_example_math_ci'
experiment_name='1129a10'
# 工程参数
gen_micro_batch_size=512 # use_dynamic_bsz=True时仍然生效
infer_micro_batch_size=512 # use_dynamic_bsz=True时不生效
train_micro_batch_size=64 # use_dynamic_bsz=True时不生效
use_dynamic_bsz=True
actor_ppo_max_token_len=3072
critic_ppo_max_token_len=3072
infer_ppo_max_token_len=3072
actor_sp_size=2
critic_sp_size=2
ref_sp_size=1
reward_sp_size=1
fsdp_size=8
xperf_tp_size=2
offload_train_memory=True
act_offload=True

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
    actor_rollout_ref.actor.optim.type=${optimizer_type} \
    actor_rollout_ref.actor.optim.force_bfloat16_state=${force_bfloat16_state} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=256 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.actor.act_offload=${act_offload} \
    actor_rollout_ref.actor.scale_pg_by_kl=False \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    trainer.remote_rm_type='grm' \
    trainer.use_remote_rm=True \
    reward_model.grm.use_grm_reverse=${use_grm_reverse} \
    reward_model.grm.score_merger=v1 \
    reward_model.rm_server.llm_serving_psm=${rm_psm} \
    reward_model.rm_server.llm_serving_idc=${rm_idc} \
    reward_model.rm_server.llm_serving_cluster=${rm_cluster} \
    reward_model.rm_server.model_name=${rm_model_name} \
    reward_model.rm_server.client_pool_size=16 \
    reward_model.rm_server.ray_actor_pool_size=16 \
    reward_model.grm.max_prompt_length=28672 \
    reward_model.grm.max_response_length=8192 \
    reward_model.grm.system_prompt_list=["'你是一个专为数学题目的回答进行评分的评分助手。给定题目和已知标准答案作评分参考，你需要对一个针对该题目的详细回答进行评分，该回答分为思考过程和正式作答两部分，思考过程是夹在<think>和</think>之间的内容（可能为空），正式作答是</think>之后的内容。你需要严格按照以下评分思路和评分步骤对回答进行评分：\n# 评分思路\n- 回答需要保证**完整性**，即回答是否完整包含思考过程和正式作答两部分，不存在思考过程过长导致丢失正式作答的情况（表现为没有出现</think>字符及之后的内容）或正式作答未能得出明确最终结论数字的情况。若回答不完整，则总得分无需考虑其他维度直接0分。\n- 在回答完整性无问题情况下，回答需要满足**准确性**，即思考过程和正式作答是否得出与题目给定标准答案一致的结果（标准答案保证一定是准确的，如果回答得出与标准答案不一致的结果则必定错误）。如准确性有误，则总得分无需考虑其他维度直接0分。\n- 在回答完整性与准确性无问题情况下，思考过程部分需额外考虑以下评分维度\n1. **思考过程简洁性**：每个推理步骤是否直接服务于问题解决，不存在冗余表述（与题目推理无关的表述）、重复思考（上文已经做过相同逻辑的思考）或无效思考（对得出结论没有直接帮助的思考）。完全不存在上述问题可得满分1分，存在1处轻微问题得0.5分，存在两处及以上即得最低分0分。\n- 在回答完整性与准确性无问题情况下，正式作答部分需额外考虑以下评分维度\n1. **正式作答便捷性**：正式作答中的解题步骤是否清晰易懂，是否以用户友好的markdown格式展示且结构层次分明；满足即可得满分1分，存在轻微瑕疵得0.5分，存在较大排版问题导致结构混乱难以理解得0分。\n2. **正式作答必要性**：正式作答中是否只包含解题必需的步骤，没有多余解释和旁白；完全不存在问题得1分，存在1处轻微问题得0.5分，否则得0分。\n# 评分步骤\n1. **检查回答完整性**：回答完整可继续步骤2，否则整体得分为0，跳到步骤5。\n2. **检查回答准确性**：首先复述题目直接给定的标准答案，然后提取出回答中经过详细推理后得出的结论数字，比较两者是否一致以确认准确性。准确性无误可继续步骤3和4，否则整体得分为0，跳到步骤5。\n3. **检查思考过程简洁性**：若思考过程为空，这一步直接得满分1分，继续步骤4；若思考过程不为空，分析简洁性维度是否存在问题并根据问题的严重程度打0、0.5或1分。\n4. **检查正式作答便捷性和必要性**：再次校验回答的完整性，若回答不完整导致不存在正式作答内容（</think>及之后的内容）或正式作答未能得出结论数字，则得0分；若完整性无问题，分析正式作答在必要性、便捷性两个维度是否存在问题并根据问题的严重程度分别打0、0.5或1分。\n5. **总结最终评分**：若回答完整性和准确性存在问题，总得分为0分；若回答完整性和准确性无问题，总得分=思考过程简洁性*60%+正式作答便捷性*20%+正式作答必要性*20%，范围为0-1分。\n6. **输出评分结论**：严格按照“回答总得分：x分”的格式输出最终评分结论。\n# 重点注意事项\n1. 你的任务是针对给定回答的思考过程和正式作答进行评分，禁止重新生成、续写和修改回答内容。\n2. 严格按照评分步骤中的6个步骤的形式进行思考和组织回复。\n3. 回答的完整性和准确性非常重要，你需要确保这两个环节的校验准确无误。\n4. 你需要确保在回复的末尾按照指定的格式进行最终评分结论的输出。'"] \
    reward_model.grm.response_postprocess_mode=[0] \
    reward_model.grm.prepare_grm_prompt_mode=[0] \
    reward_model.grm.score_parser=['v2'] \
    reward_model.grm.score_weight=[1.0] \
    reward_model.grm.empty_response_default_score=[0.0] \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_max_token_len=${critic_ppo_max_token_len} \
    critic.optim.type=${optimizer_type} \
    critic.optim.force_bfloat16_state=${force_bfloat16_state} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.infer_micro_batch_size=${infer_micro_batch_size} \
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
    reward_model.add_int_verify=False \
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
    trainer.eval_before_training=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps="disable" \
    +actor_rollout_ref.rollout.complete_ratio=1.0 \
    +actor_rollout_ref.rollout.max_off_policy_steps=5 \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    critic.fsdp_size=${fsdp_size} \
    reward_model.fsdp_size=${fsdp_size} \
    critic.act_offload=${act_offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.rollout.mode=server \
    data.shuffle=False \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${xperf_tp_size} \
    +actor_rollout_ref.rollout.use_vllm=True \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    trainer.offload_train_memory=${offload_train_memory} \
    trainer.total_steps=${NUM_STEPS} \
    # trainer.save_train_batch_dir=${default_hdfs_dir}/train_batch \
    # trainer.load_train_batch_path=${default_hdfs_dir}/train_batch/train_batch_1.pt