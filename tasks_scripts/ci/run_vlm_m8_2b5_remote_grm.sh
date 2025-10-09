set -x
ray stop --force
export THINK_TEMPLATE=v1
export NCCL_DEBUG=WARN

# ckpt和路径
SFT_MODEL_PATH=hdfs://harunawl/home/byte_data_seed_wl/vlm/iccv/user/lingyue/checkpoints/xperf/m8_vlm_2b5_vit_600m_vlmct_longct_cp2_v7.0.7_cotv12_fix_128k_h800_mixstage_regress_new2
TRAIN_FILE=hdfs://harunawl/home/byte_data_seed_wl/vlm/iccv/user/yeqinghao/rl_data/250827_qrl_v408_m8v63_final_woGUI_8k_3307843_ASFormat/00_chunk_001.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/data/rl/eval_alphaseed_mathvision_fix_v2_dot.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/test/vlm_grpo

# tracking实验名
project_name='alpha_seed_vlm'
experiment_name="m8_2b5_grpo_8k_train_func_call_zerobench_v_star_train_91"

default_hdfs_dir=hdfs://haruna/home/byte_data_seed/hl_lf/user/xiangyongan/VLM/tmp/${project_name}/${experiment_name}
save_train_batch_dir=hdfs://haruna/home/byte_data_seed/hl_lf/user/xiangyongan/VLM/tmp/${project_name}/${experiment_name}/batch_data
load_train_batch_path=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/alpha_seed_vlm/m8_2b5_grpo_8k_train_func_call_zerobench_v_star_train_1/batch_data/train_batch_1.pt
echo "default_hdfs_dir ${default_hdfs_dir}"

# 训练长度
max_prompt_length=8192
max_response_length=16384
max_grm_prompt_length=65536
# batch size && 训练epoch
train_batch_size=8
ppo_mini_batch_size=8
val_batch_size=8
total_epochs=5
test_freq=100
save_freq=100
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
num_bon=1
bon_strategy=all
kl_penalty=low_var_kl
temperature=1.2

# tracking实验名
project_name='alpha_seed_vlm'
experiment_name="m8_2b5_grpo_$(date +%F)"
# 工程参数
gen_micro_batch_size=4 # use_dynamic_bsz=True时仍然生效
infer_micro_batch_size=4 # use_dynamic_bsz=True时不生效
train_micro_batch_size=4 # use_dynamic_bsz=True时不生效

use_dynamic_bsz=True
actor_ppo_max_token_len=36864
critic_ppo_max_token_len=36864
infer_ppo_max_token_len=36864
# actor_sp_size=2
# critic_sp_size=2

remote_rm_type='grm'
use_remote_rm=True
use_rm_reverse=True
rm_psm="data.aml.arnold_inference_59310350"
rm_idc="'lq,lf,hl,yg,gl,wlby'"
rm_cluster="default"
rm_model_name="vlm/M8-23B-MoE-W8A8-GRM_G1_T408_step275_prompt24k_resp8k-S200"
###

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
    data.dist_image=True \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=4 \
    data.truncation='left' \
    +data.chat_template=seed \
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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.actor.clip_ratio_high=0.15 \
    actor_rollout_ref.actor.clip_ratio_low=0.1 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.rollout.enable_paged_attention=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.scale_pg_by_kl=False \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    algorithm.adv_estimator=${adv_estimator} \
    reward_model.need_punish_trunc=True \
    reward_model.trunc_punish_score=-0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
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
    trainer.remote_rm_type=${remote_rm_type} \
    trainer.use_remote_rm=${use_remote_rm} \
    reward_model.grm.use_rm_reverse=${use_rm_reverse} \
    reward_model.grm.score_merger=v1 \
    reward_model.rm_server.llm_serving_psm=${rm_psm} \
    reward_model.rm_server.llm_serving_idc=${rm_idc} \
    reward_model.rm_server.llm_serving_cluster=${rm_cluster} \
    reward_model.rm_server.model_name=${rm_model_name} \
    reward_model.rm_server.client_pool_size=16 \
    reward_model.rm_server.ray_actor_pool_size=16 \
    actor_rollout_ref.rollout.mode=server \
    reward_model.grm.system_prompt_list=["'你是一个专为数学题目的回答进行评分的评分助手。给定题目和已知标准答案作评分参考，你需要对一个针对该题目的详细回答进行评分，该回答分为思考过程和正式作答两部分，思考过程是夹在<think>和</think>之间的内容（可能为空），正式作答是</think>之后的内容。你需要严格按照以下评分思路和评分步骤对回答进行评分：\n# 评分思路\n- 回答需要保证**完整性**，即回答是否完整包含思考过程和正式作答两部分，不存在思考过程过长导致丢失正式作答的情况（表现为没有出现</think>字符及之后的内容）或正式作答未能得出明确最终结论数字的情况。若回答不完整，则总得分无需考虑其他维度直接0分。\n- 在回答完整性无问题情况下，回答需要满足**准确性**，即思考过程和正式作答是否得出与题目给定标准答案一致的结果（标准答案保证一定是准确的，如果回答得出与标准答案不一致的结果则必定错误）。如准确性有误，则总得分无需考虑其他维度直接0分。\n- 在回答完整性与准确性无问题情况下，思考过程部分需额外考虑以下评分维度\n1. **思考过程简洁性**：每个推理步骤是否直接服务于问题解决，不存在冗余表述（与题目推理无关的表述）、重复思考（上文已经做过相同逻辑的思考）或无效思考（对得出结论没有直接帮助的思考）。完全不存在上述问题可得满分1分，存在1处轻微问题得0.5分，存在两处及以上即得最低分0分。\n- 在回答完整性与准确性无问题情况下，正式作答部分需额外考虑以下评分维度\n1. **正式作答便捷性**：正式作答中的解题步骤是否清晰易懂，是否以用户友好的markdown格式展示且结构层次分明；满足即可得满分1分，存在轻微瑕疵得0.5分，存在较大排版问题导致结构混乱难以理解得0分。\n2. **正式作答必要性**：正式作答中是否只包含解题必需的步骤，没有多余解释和旁白；完全不存在问题得1分，存在1处轻微问题得0.5分，否则得0分。\n# 评分步骤\n1. **检查回答完整性**：回答完整可继续步骤2，否则整体得分为0，跳到步骤5。\n2. **检查回答准确性**：首先复述题目直接给定的标准答案，然后提取出回答中经过详细推理后得出的结论数字，比较两者是否一致以确认准确性。准确性无误可继续步骤3和4，否则整体得分为0，跳到步骤5。\n3. **检查思考过程简洁性**：若思考过程为空，这一步直接得满分1分，继续步骤4；若思考过程不为空，分析简洁性维度是否存在问题并根据问题的严重程度打0、0.5或1分。\n4. **检查正式作答便捷性和必要性**：再次校验回答的完整性，若回答不完整导致不存在正式作答内容（</think>及之后的内容）或正式作答未能得出结论数字，则得0分；若完整性无问题，分析正式作答在必要性、便捷性两个维度是否存在问题并根据问题的严重程度分别打0、0.5或1分。\n5. **总结最终评分**：若回答完整性和准确性存在问题，总得分为0分；若回答完整性和准确性无问题，总得分=思考过程简洁性*60%+正式作答便捷性*20%+正式作答必要性*20%，范围为0-1分。\n6. **输出评分结论**：严格按照“回答总得分：x分”的格式输出最终评分结论。\n# 重点注意事项\n1. 你的任务是针对给定回答的思考过程和正式作答进行评分，禁止重新生成、续写和修改回答内容。\n2. 严格按照评分步骤中的6个步骤的形式进行思考和组织回复。\n3. 回答的完整性和准确性非常重要，你需要确保这两个环节的校验准确无误。\n4. 你需要确保在回复的末尾按照指定的格式进行最终评分结论的输出。'"] \
    reward_model.grm.response_postprocess_mode=[0] \
    reward_model.grm.prepare_grm_prompt_mode=[0] \
    reward_model.grm.score_parser=['v2'] \
    reward_model.grm.score_weight=[1.0] \
    reward_model.grm.empty_response_default_score=[0.0] \
    +actor_rollout_ref.rollout.complete_ratio=1.0 \
    +actor_rollout_ref.rollout.max_off_policy_steps=0 \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${xperf_tp_size} \
    trainer.offload_train_memory=${offload_train_memory} \
    critic.profile.enable=False \
    actor_rollout_ref.actor.profile.enable=False \
    trainer.volc_ark_key="f2b9d02c-4fd5-4a3c-a18f-ffaa806c1f64" \
    tasks.reward_manager=tasks.vlm.reward_manager.VLMRewardManager \
    tasks.trainer=tasks.vlm.ppo_trainer.VLMRayPPOTrainer \
    trainer.volc_model_name="ep-20250523002206-rn5sm" \
    reward_model.grm.max_prompt_length=${max_grm_prompt_length} \
    reward_model.grm.max_response_length=8192 \
    tasks.dataset=alpha_seed.utils.dataset.mix_rl_dataset.MixRLDataset \
    data.think_template=v3 \
    +ext=vlm_ext \
    rollout_server.handler="vlm/single_turn" \
    data.check_template=False