set -x
export CUDA_LAUNCH_BLCOKING=True
NUM_STEPS="${NUM_STEPS:-240}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
echo $NUM_STEPS

# ckpt和路径

SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/m8_vlm_680m_seedvit
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/xiaoboqin/data/rlhf/math/mmathcot_v4_hard_w_sys_for_rl.parquet
TRAIN_FILE=hdfs://harunawl/home/byte_data_seed_wl/vlm/iccv/user/yeqinghao/rl_data/250827_qrl_v408_m8v63_final_woGUI_8k_3307843_ASFormat/00_chunk_000.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/eval_mathvision_mini.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/test/vlm_grpo


# 训练长度
max_prompt_length=8192
max_response_length=1024
# batch size && 训练epoch

train_batch_size=8
ppo_mini_batch_size=8
val_batch_size=8
total_epochs=200
test_freq=-1
save_freq=-1
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
num_bon=2
bon_strategy=all
kl_penalty=low_var_kl
temperature=1.2

# tracking实验名
project_name='alpha_seed_vlm'
experiment_name="m8_2b5_grpo_$(date +%F)"
# 工程参数
gen_micro_batch_size=8 # use_dynamic_bsz=True时仍然生效
infer_micro_batch_size=8 # use_dynamic_bsz=True时不生效
train_micro_batch_size=8 # use_dynamic_bsz=True时不生效

use_dynamic_bsz=True
actor_ppo_max_token_len=36864
critic_ppo_max_token_len=36864
infer_ppo_max_token_len=36864

actor_sp_size=1
critic_sp_size=1
ref_sp_size=1
reward_sp_size=1
fsdp_size=-1
xperf_tp_size=2
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
    data.truncation='error' \
    +data.chat_template=seed \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_token_len=${infer_ppo_max_token_len} \
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
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.actor.scale_pg_by_kl=False \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
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
    +actor_rollout_ref.rollout.complete_ratio=1.0 \
    +actor_rollout_ref.rollout.max_off_policy_steps=0 \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    reward_model.need_punish_trunc=True \
    reward_model.trunc_punish_score=-0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${xperf_tp_size} \
    +actor_rollout_ref.rollout.use_vllm=False \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    trainer.offload_train_memory=${offload_train_memory} \
    critic.profile.enable=False \
    critic.profile.upload_to_mlx=False \
    critic.profile.filename=actor.tp${xperf_tp_size}.fsdp${fsdp_size} \
    actor_rollout_ref.actor.profile.enable=False \
    actor_rollout_ref.actor.profile.upload_to_mlx=False \
    actor_rollout_ref.actor.profile.filename=actor.tp${xperf_tp_size}.fsdp${fsdp_size} \
    trainer.total_steps=${NUM_STEPS} \
    trainer.save_cases_to_hdfs=False \
    data.think_template=v3 \
    tasks.dataset=alpha_seed.utils.dataset.mix_rl_dataset.MixRLDataset \
    actor_rollout_ref.rollout.mode=server \
    tasks.reward_manager=tasks.vlm.reward_manager.VLMRewardManager \
    tasks.trainer=tasks.vlm.ppo_trainer.VLMRayPPOTrainer \
    data.add_thinking_prompt=True \
    data.add_thinking_prompt_ratio=0.95 \
    data.override_ability_with_datasource=False \
    data.override_datasource_with_ability=True \
    data.ability_list=\"NLP,code,collie_supply,cosplay,creation,game,instrruler_cl,instrruler_if,instrruler_logic,instrruler_multiround,instrruler_nologic,instrruler_word_cnt_cn,instrruler_word_cnt_en,knowledge,logic,plugin,reasoning,sandbox_code,subject,tob,tob_intelligent_surveillance,tob_liubin_new_add,unknown,verifiable_function_call,verifier_VisualPuzzle,verifier_alphaseed_CHE,verifier_alphaseed_MATH,verifier_alphaseed_PHY,verifier_alphaseed_STEM,verifier_chart,verifier_code_sandbox,verifier_complex_basic_perception_manual_annotation,verifier_critic,verifier_edu_hard,verifier_grounding_complex,verifier_longdoc,verifier_math,verifier_math_gauth,verifier_math_global_data,verifier_math_vlm_stem,verifier_multichoiceFix,verifier_multichoiceFix_MathvisionStyle,verifier_service,verifier_stem_gauth,verifier_stem_global_data,verifier_temporal_ground,verifier_video_shuffle,verifier_visual_functionCall_answer_geoguess_combine,verifier_vlm_stem_service,verifier_vlm_visionarena,vlm_complex_instruction,vlm_doubao_app,vlm_inhouse_complex,vlm_inhouse_simple,vlm_knowledge,vlm_mm_OCR,vlm_mm_complex,vlm_mm_knowledge,vlm_mm_simple\" \
    data.ability_kl_weights=\"1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0\" \
    data.rm_required_abilities=\"NLP,code,collie_supply,cosplay,creation,game,instrruler_cl,instrruler_if,instrruler_logic,instrruler_multiround,instrruler_nologic,instrruler_word_cnt_cn,instrruler_word_cnt_en,knowledge,logic,plugin,reasoning,subject,tob,tob_intelligent_surveillance,tob_liubin_new_add,unknown,verifiable_function_call,verifier_vlm_visionarena,vlm_complex_instruction,vlm_doubao_app,vlm_inhouse_complex,vlm_inhouse_simple,vlm_knowledge,vlm_mm_OCR,vlm_mm_complex,vlm_mm_knowledge,vlm_mm_simple\" \
    "data.special_tokens.think_begin='<think_never_used_51bce0c785ca2f68081bfa7d91973934>'" \
    "data.special_tokens.think_end='</think_never_used_51bce0c785ca2f68081bfa7d91973934>'" \
    +ext=vlm_ext \
    data.check_template=False \
    trainer.remote_rm_type=grm \
    reward_model.grm.system_prompt_list=["'You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, i.e. <think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>system\n你是一个专业的AI助手评估专家，请按照以下步骤评估AI回答1和回答2的优劣。\n\n第一步：提供判断原则\n首先逐句仔细研读题目内容，基于题目自身的要求和特点，独立思考并制定合理的判断准则。可参考通用判断标准作为思路启发，但需结合题目具体情况进行个性化调整，避免直接套用模板。\n每条判断准则需包含三个核心要素：\n1. 原则名称：用简洁准确的语言概括准则核心内容\n2. 权重分配：以百分比形式（0%-100%）标注该准则在整体评分中的重要程度\n3. 评分说明：详细说明0-4分五个评分档位（每档1分递增）的具体评分标准，需结合题目要求明确各档位的区分要点。\n注意：\n1. 拆解判断原则时，应从多维度进行系统化细分，确保覆盖所有关键要素且细节完整详实，以便为后续的评分、对比提供更容易判别的依据。\n2. 思考评价标准的时候要注意，回答的质量和回复的长度、回复个数没有正相关关系，不是越长、越多细节和例子越好，好的回答应注意信息的平衡，重点突出、详略得当、逻辑清晰、信息真实，避免过度冗余、晦涩、复杂的信息影响用户的理解。\n3. 思考评价创作类问题文采标准的时候要注意：创作中的文采并非越多细节、越多晦涩意象堆叠、华丽词藻堆叠越好；真正的文采既有形式上的优美，又有内容上的深度，能够打动人心并留下持久的印象。\n\n第二步：逐项评分\n对每个回答按以下流程评分：\n1. 说明该维度的评分理由\n2. 给出 0-4 分的整数评分（例如4 = 好用，3 = 可用，2 = 有点用，1=难用，0=没用）\n3. 计算加权得分（评分 × 权重）\n\n第三步：总分对比\n单项得分 = 分数 × 权重\n总分 = 各单项得分之和，总分在0～4分之间\n**严格按此格式书写总分**:\n回答1总得分：$x分\n回答2总得分：$y分\n\n第四步：结论输出\n**严格按此格式书写结论**：\n"回答1对比回答2落败" 或 "回答1对比回答2胜出"\n\n重点说明：\n若出现平分（如总分相同），需额外说明决胜依据\n必须验证计算结果\n需明确指出错误类型\n\n<通用判断标准>\n1. 以用户视角的体验为核心，即考虑当用户看到模型回复时，会产生什么样的体验，5档打分就是基于该体验满意度的度量值。\n2. 用户体验依赖主要需求被满足的情况，即在判断某个子维度是否加分减分时，优先考虑该维度的内容是否围绕主需。\n3. 打分时，不过度拘泥于细节，不执着于“扣分思维”，要整体判断内容质量以及问题对主需的影响程度，最终给出基于用户视角的整体打分。\n4. 打分有分差必有偏序，有偏序不一定有分差（同分也可出偏序）。\n5. 整体判断逻辑是从用户的核心需求出发，不同体裁/类型的题目，用户的核心关注点不同：\n  1）知识方向：\n    a. 强答案类问题/封闭问题：用户最核心的关注点是准确性，因此模型最重要的是要“答对”，答案内容和格式无误即可4分。（在部分强答案问题中，如果答案正确，3分和4分价值相等，直接使用4分；如有问题，再从4分根据问题严重程度向下扣分）\n    b. 开放类问题：在对信息准确性要求的基础上，用户还对答案组织的逻辑、深度与广度有需求，因此需注重内容质量，蜻蜓点水式浅谈只能3分，条理清晰、阐述充分才能4分。\n  2）创作方向：用户的高优需求是内容质量，核心关注点是“写好”，对内容的丰富度、文采、深度、特异性有更高要求，因此内容没有问题但质量一般是3分，有增益可升至4分。\n\n主要概念说明：\n主要理念：主需\n说明：主需是指用户指令中需要模型完成的核心动作，或用户需要模型提供的核心信息。识别及满足主需是模型最重要的任务，享有最高优先级。如内容、体裁、主要指令\n\n主要理念：次需\n说明：次需多为指令中的约束条件，例如字数、风格、格式等，或主需以外的扩展类需求等。次需的优先级因体裁和场景而异：\n- 在创作类需求中，内容价值、风格完成度的优先级高于字数满足度；\n- 在ToB等强格式需求中，格式问题可能影响批量信息读取，格式的优先级提升。\n\n主要理念：丰富度\n说明：判断内容是否丰富，不能仅看维度或列举条数的多寡，而要综合判断内容质量、多样化程度、主需满足度。\n\n主要理念：增益与冗余\n说明：- 基于主需的延展信息，有明显收益才会加分，有明显冗余（如与主需关联度差、篇幅过大等）才会扣分；\n- 无功无过可不加分不扣分，避免扩大化执行，影响生成内容多样性。\n- 判断冗余需考虑用户体验，例如不相关内容若出现在主需前面，用户会有迟迟无法进入主题之感；若出现在主需后面，用户的感知程度便会减弱。\n\n常见问题类型：\n\n需求识别：对主需的识别不对\n常见bad case举例：误认为一些prompt包含安全风险而拒绝回答、需求写诗歌，但实际回答是一篇小说\n\nCodeswitch：在非英文场景生硬的插入英文，中英混杂\n\n基础相关性：与主需不相关\n常见bad case举例：问A给A+B\n\n信息缺失：缺失关键/核心要点，明显影响主需\n\n信息错误：信息错误，无法满足主需\n\n内容重复：段落重复、句子重复、句式内容重复、轮次间重复、用词重复等影响用户观感的结果\n\n信息丰富度差\n常见bad case举例：比如“珠穆朗玛峰”仅给到高度\n\n便捷性差：排版混乱，答案在文中不易找等\n\n内容价值低：内容完成度低\n常见bad case举例：笑话不好笑\n\n文本瑕疵：存在明显语病、错别字、标点混乱、表述不当等问题，影响内容理解\n\n对话能力差：有对话感，存在机械感\n\n逻辑表达差：存在局部错误、矛盾或前后不一致\n\n多轮错误：上下文理解错误，导致本轮没有满足主需\n常见bad case举例：多轮场景下，遗忘上文内容，多出现于长篇幅对话\n\n真实性：信息不真实\n常见bad case举例：-1993年是闰年(真实性有误）\n-三角形两边之和小于第三遍（真实性有误）\n-马斯克是特斯拉创始人（争议问题，不算错误）\n\n对话幻觉：在对话场景中，无中生有行为\n</通用判断标准>'"] \
    reward_model.grm.response_postprocess_mode=[3] \
    reward_model.grm.prepare_grm_prompt_mode=[0] \
    reward_model.grm.score_parser=['v3'] \
    reward_model.grm.score_weight=[1.0] \
    reward_model.grm.empty_response_default_score=[0.0] \
    rollout_server.handler="vlm/single_turn"
