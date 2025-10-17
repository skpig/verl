import json
import logging
import os

import ray

from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import filter_thinking_part, match_visual_cot_format
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import ExtractAnswerFailed, VerifierFailed
from alpha_seed.utils.reward_score.utils import Verifier
from alpha_seed.utils.reward_score import response_post_proc

logger = logging.getLogger(__file__)


def post_process_solution_str(config, solution_str, eos_token):
    last_eos_idx = solution_str.rfind(eos_token)
    if last_eos_idx >= 0:
        eos_then_assistant = f'{eos_token}<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n'
        # When use_remote_verifier=True, the solution_str here hasn't been forcibly appended EOS yet.
        # This if-statement is for avoiding accidentally rsplit the EOS **before** the assistant response:
        if solution_str[last_eos_idx:last_eos_idx + len(eos_then_assistant)] != eos_then_assistant:
            solution_str = solution_str.rsplit(eos_token, 1)[0]  # Remove the EOS **after** the assistant response.

    if config.reward_model.use_last_response == 'summarize':
        solution_str_post_proc = response_post_proc.summary_postprocess(
            solution_str,
            last_response_sep=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict)
    elif config.reward_model.use_last_response == 'lastcodeblock':
        solution_str_post_proc = response_post_proc.last_codeblock_postprocess(
            solution_str,
            codeblock_seps=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict)
    else:
        solution_str_post_proc = solution_str
    return solution_str_post_proc


class VLMRouter(Verifier, reward_style="vlm_verifier_router"):

    def is_remote(self):
        return True

    def preprocess(self, *args, **kwargs):
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
            input_ids = [x for x in input_ids if x != -100]
            solution_str = self.tokenizer.decode(input_ids)
            solution_str_post_proc = post_process_solution_str(self.config,
                                                               solution_str,
                                                               eos_token=self.tokenizer.eos_token)
            ## we need only the response part for vlm_verifier_router
            marker_user = '<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>user\n'
            marker_assistant = '<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n'

            assert marker_assistant in solution_str_post_proc and marker_user in solution_str_post_proc, f"marker_assistant {marker_assistant} or marker_user {marker_user} not in solution_str_post_proc {solution_str_post_proc}"
            solution_str_post_proc_anwswer = solution_str_post_proc.rsplit(marker_user, 1)[1]
            solution_str_post_proc_anwswer = solution_str_post_proc_anwswer.split(marker_assistant, 1)[1]
        else:
            solution_str_post_proc_anwswer = kwargs['solution_str']
        ground_truth = kwargs['ground_truth']
        think_template = self.config.data.think_template if self.config.data.think_template is not None else 'v2'
        no_thinking_required = kwargs.get('no_thinking_required', False)
        return solution_str_post_proc_anwswer, ground_truth, think_template, self.config.trainer.code_sandbox_psm, \
            self.config.trainer.volc_ark_key, self.config.trainer.volc_model_name, no_thinking_required

    @staticmethod
    def compute_score(solution_str, ground_truth, think_template, code_sandbox_psm, volc_ark_key, volc_model_name,
                      no_thinking_required, **argv) -> float:
        return compute_score(solution_str, ground_truth, code_sandbox_psm, volc_ark_key, volc_model_name,
                             think_template, no_thinking_required)


# Note: AlphaSeed uses [-1, +1] for reward scores and -2 for error handling, which is different from SeedRL's [0, 1].
def compute_score(solution_str,
                  ground_truth,
                  code_sandbox_psm: str,
                  volc_ark_key: str,
                  volc_model_name: str,
                  think_template: str,
                  no_thinking_required: bool = False,
                  **kwargs) -> float:
    try:
        ref = submit_verifier.remote(solution_str, ground_truth, code_sandbox_psm, volc_ark_key, volc_model_name,
                                     think_template, no_thinking_required)
        try:
            score = ray.get(ref, timeout=1500)
        except ray.exceptions.GetTimeoutError:
            # timeout
            score = -2
            logger.warning(
                f"[VLM VERIFIER ERROR] compute_score_client got timeout error, with ground_truth={ground_truth}")
            ray.cancel(ref, force=True, recursive=True)
    except Exception as e:
        # other error
        logger.warning(
            f"[VLM VERIFIER ERROR] compute_score_client got unknown error: {e}, with ground_truth={ground_truth}")
        score = -2
    return score


@ray.remote
def submit_verifier(response: str,
                    verifier_feature: str,
                    code_sandbox_psm: str,
                    volc_ark_key: str,
                    volc_model_name: str,
                    think_template: str,
                    no_thinking_required: bool = False):
    os.environ['THINK_TEMPLATE'] = think_template
    full_rollout = response

    if not response:
        raise ValueError('Empty response!')
    if not verifier_feature:
        raise ValueError('Empty verifier_feature!')

    feature: dict = json.loads(verifier_feature)
    answer: str = feature.get('answer', '')
    verifier_name: str = feature['verifier_name']

    bos_token = "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
    bos_assistant_nl = f'{bos_token}assistant\n'
    bos_user_nl = f'{bos_token}user\n'
    eos_token = "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"

    assert bos_user_nl not in response, 'Assuming `response` to have only decoding rollout and no input context.'
    assert not response.startswith(bos_assistant_nl), 'Leading `[BOS]assistant` belongs to input context, not rollout.'
    assert not response.endswith(eos_token), 'The ending [EOS] should have been removed by post_process_solution_str.'
    if response.startswith(bos_token):
        print(f'[WARNING] The model seems to be generating [BOS]! {repr(full_rollout)}')
        response = ''

    # Ensure that the last turn is from role="assistant", not from other roles such as role="tool":
    k = response.rfind(bos_token)
    while (k >= 0) and (response[k:k + len(bos_assistant_nl)] != bos_assistant_nl):
        response = response[:k]
        if not response.endswith(eos_token):
            print(f'[WARNING] Missing [EOS] before [BOS]! Is this [BOS] generated by the model? {repr(full_rollout)}')
            response = ''
            break
        response = response[:-len(eos_token)]
        k = response.rfind(bos_token)

    validation_verifiers = [
        'verifier_vstar',
        'verifier_zerobench',
        'verifier_charxiv',
    ]
    # Verify if it follows the VisualCoT format. Allow the last turn to be FC to support LLM FC data.
    if (verifier_name not in validation_verifiers) and (os.getenv("THINK_TEMPLATE", "v2") != "v1"):
        # Skip these two for now. May delete this `if` in the future.
        if verifier_name in ["text_function_call_v3"]:
            only_check_think_format = True
        else:
            only_check_think_format = False

        if not match_visual_cot_format(
                response,
                verifier_feature=feature,
                allow_last_turn_fc=True,
                no_thinking_required=no_thinking_required,
                only_check_think_format=only_check_think_format,
        ):
            response = ''  # Let it fail.

    verify_full_rollout = feature.get('verify_full_rollout', False) or (verifier_name in (
        'visual_cot_verifier',
        'auxline_rule_verifier',
        'video_grounding_counting_verifier',
        'rotate_tool_verifier',
        'visual_chained_tool_use_verifier',
    ))
    if not verify_full_rollout:
        # Discard turns related with function calling and keep only the last assistant response:
        k = response.rfind(bos_assistant_nl)
        if k >= 0:
            response = response[k + len(bos_assistant_nl):]
        # Discard the CoT part:
        response, _ = filter_thinking_part(response, no_thinking_required=no_thinking_required)

    try:
        if response == '':
            raise ExtractAnswerFailed('Failed basic format checks!')

        # Notice for pip dependencies:
        #   - Verifier "count/pointing/bbox/countbypoint/mcqa/action_count/temporal_ground" requires: shapely==2.0.6 word2number==1.1
        #   - Verifier "gui" requires: jieba==0.42.1 rouge_chinese==1.0.3
        if verifier_name == 'math':
            from alpha_seed.utils.reward_score.vlm_verifiers.math_verifier import MathVerifier
            result = MathVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'math_v2':
            from alpha_seed.utils.reward_score.vlm_verifiers.math_verifier import MathV2Verifier
            result = MathV2Verifier(volc_ark_key=volc_ark_key,
                                    volc_model_name=volc_model_name).verify(response=response,
                                                                            verifier_feature_dict=feature)
        elif verifier_name == 'arena_code_switch_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.arena_lang_llm_verifier import LLMArenaLangVerifier
            result = LLMArenaLangVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'math_verifier_service':
            from alpha_seed.utils.reward_score.vlm_verifiers.math_verifier import ModelBasedMathVerifierVolc
            result = ModelBasedMathVerifierVolc(volc_ark_key=volc_ark_key,
                                                volc_model_name=volc_model_name).verify(response=response,
                                                                                        verifier_feature_dict=feature)
        elif verifier_name == 'stem_verifier_service':
            from alpha_seed.utils.reward_score.vlm_verifiers.stem_verifier import ModelBasedStemVerifierVolc
            result = ModelBasedStemVerifierVolc(volc_ark_key=volc_ark_key,
                                                volc_model_name=volc_model_name).verify(response=response,
                                                                                        verifier_feature_dict=feature)
        elif verifier_name == 'puzzle_verifier_service':
            from alpha_seed.utils.reward_score.vlm_verifiers.puzzle_verifier import ModelBasedPuzzleVerifierVolc
            result = ModelBasedPuzzleVerifierVolc(volc_ark_key=volc_ark_key,
                                                  volc_model_name=volc_model_name).verify(response=response,
                                                                                          verifier_feature_dict=feature)
        elif verifier_name == 'basic_perception_verifier_service':
            from alpha_seed.utils.reward_score.vlm_verifiers.basic_perception_verifier import ModelBasedPerceptionVerifierVolc
            result = ModelBasedPerceptionVerifierVolc(
                volc_ark_key=volc_ark_key, volc_model_name=volc_model_name).verify(response=response,
                                                                                   verifier_feature_dict=feature)
        elif verifier_name == 'code_sandbox':
            from alpha_seed.utils.reward_score.vlm_verifiers.code_sandbox_verifier import CodeSandboxVerifier
            result = CodeSandboxVerifier(code_sandbox_service_psm=code_sandbox_psm).verify(
                response=response, verifier_feature_dict=feature)
        elif verifier_name == 'boxed_str':
            from alpha_seed.utils.reward_score.vlm_verifiers.string_verifier import BoxStrVerifier
            result = BoxStrVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "count":
            from alpha_seed.utils.reward_score.vlm_verifiers.count_verifier import CountVerifier
            result = CountVerifier().verify(response=response, verifier_feature_dict=feature, delta=0.0)
        elif verifier_name == "pointing":
            from alpha_seed.utils.reward_score.vlm_verifiers.point_verifier import PointVerifier
            result = PointVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "bbox":
            from alpha_seed.utils.reward_score.vlm_verifiers.bbox_verifier import BBoxVerifier
            result = BBoxVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "countbypoint":
            from alpha_seed.utils.reward_score.vlm_verifiers.cotcount_verifier import CoTCountVerifier
            result = CoTCountVerifier().verify(response=response, verifier_feature_dict=feature, delta=0.6)
        elif verifier_name == 'plain_str':
            from alpha_seed.utils.reward_score.vlm_verifiers.string_verifier import PlainStrVerifier
            result = PlainStrVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "findiff":
            from alpha_seed.utils.reward_score.vlm_verifiers.finddiff_verifier import FindDiffVerifier
            result = FindDiffVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "finddiffreflect":
            from alpha_seed.utils.reward_score.vlm_verifiers.finddiff_verifier import FindDiffReflectVerifier
            result = FindDiffReflectVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "maze":
            from alpha_seed.utils.reward_score.vlm_verifiers.maze_verifier import MazeVerifier
            result = MazeVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "matching_game":
            from alpha_seed.utils.reward_score.vlm_verifiers.matching_game_verifier import MatchingGameVerifier
            result = MatchingGameVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "gui":
            from alpha_seed.utils.reward_score.vlm_verifiers.gui_verifier import GUIVerifier
            result = GUIVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "mcqa":
            from alpha_seed.utils.reward_score.vlm_verifiers.mcqa_verifier import MCQAVerifier
            result = MCQAVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "mcqa_instruct":
            from alpha_seed.utils.reward_score.vlm_verifiers.mcqa_instruct_verifier import MCQAInstructVerifier
            result = MCQAInstructVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "action_count":
            from alpha_seed.utils.reward_score.vlm_verifiers.action_count_verifier import ActionCountVerifier
            result = ActionCountVerifier().verify(response=response, verifier_feature_dict=feature, delta=0.8)
        elif verifier_name == "temporal_ground":
            from alpha_seed.utils.reward_score.vlm_verifiers.temporal_ground_verifier import TemporalGroundVerifier
            result = TemporalGroundVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "critic":
            from alpha_seed.utils.reward_score.vlm_verifiers.critic_verifier_v3 import PointwiseCriticVerifier
            result = PointwiseCriticVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "vlm_grounding_complex":
            from alpha_seed.utils.reward_score.vlm_verifiers.grounding_complex import ModelBasedGroundingComplexVerifierVolc
            result = ModelBasedGroundingComplexVerifierVolc(
                volc_ark_key=volc_ark_key, volc_model_name=volc_model_name).verify(response=response,
                                                                                   verifier_feature_dict=feature)
        elif verifier_name == 'single_bbox_llm_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.single_bbox_llm_verifier import SingleBBoxVerifier
            result = SingleBBoxVerifier(volc_ark_key=volc_ark_key,
                                        volc_model_name=volc_model_name).verify(response=response,
                                                                                verifier_feature_dict=feature)
        elif verifier_name == 'PHYSICS_benchmark_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.PHYSICS_benchmark_verifier import PhysicsVerifier
            result = PhysicsVerifier(volc_ark_key=volc_ark_key,
                                     volc_model_name=volc_model_name).verify(response=response,
                                                                             verifier_feature_dict=feature)
        elif verifier_name == 'visual_cot_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.visual_cot_verifier import VisualCoTVerifier
            result = VisualCoTVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'visual_cot_verifier_geoguess_combine':
            from alpha_seed.utils.reward_score.vlm_verifiers.visual_cot_verifier_geoguess_combine import VisualCoTVerifier_Geo_Combine
            result = VisualCoTVerifier_Geo_Combine().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'visual_cot_verifier_maze':
            from alpha_seed.utils.reward_score.vlm_verifiers.visual_cot_verifier_maze import VisualCoTVerifier_Maze
            result = VisualCoTVerifier_Maze().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'verifier_vstar':
            from alpha_seed.utils.reward_score.vlm_verifiers.validation.vstar_verifier import VstarVerifier
            result = VstarVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'verifier_zerobench':
            from alpha_seed.utils.reward_score.vlm_verifiers.validation.zerobench_verifier import ZeroBenchVerifier
            result = ZeroBenchVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'verifier_charxiv':
            from alpha_seed.utils.reward_score.vlm_verifiers.validation.charxiv_verifier import CharaxivVerifier
            result = CharaxivVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'bracket_rule_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.bracket_rule_verifier import BracketRuleVerifier
            result = BracketRuleVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'auxline_rule_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.auxline_verifier import AuxlineVerifier
            result = AuxlineVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'point_rule_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.auxline_point_verifier import AuxlinePointVerifier
            result = AuxlinePointVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'video_shuffle':
            from alpha_seed.utils.reward_score.vlm_verifiers.video_shuffle_verifier import VideoShuffleVerifier
            result = VideoShuffleVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'video_shuffle_strict':
            from alpha_seed.utils.reward_score.vlm_verifiers.video_shuffle_verifier import VideoShuffleVerifierStrict
            result = VideoShuffleVerifierStrict().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'collie_supply':
            from alpha_seed.utils.reward_score.vlm_verifiers.collie_verifier import CollieSupplyVerifier
            result = CollieSupplyVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'general_sandbox_code':
            from alpha_seed.utils.reward_score.vlm_verifiers.general_sandbox_code_verifier import GeneralSandboxVerifier
            result = GeneralSandboxVerifier(code_sandbox_service_psm=code_sandbox_psm).verify(
                response=response, verifier_feature_dict=feature)
        elif verifier_name == 'video_grounding_counting_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.video_grounding_counting import VideoGroundingCountingVerifier
            result = VideoGroundingCountingVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'rotate_tool_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.rotate_tool_verifier import RotateToolVerifier
            result = RotateToolVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'chart_verifier_service':
            from alpha_seed.utils.reward_score.vlm_verifiers.chart_verifier import ModelBasedChartVerifierVolc
            result = ModelBasedChartVerifierVolc(volc_ark_key=volc_ark_key,
                                                 volc_model_name=volc_model_name).verify(response=response,
                                                                                         verifier_feature_dict=feature)
        elif verifier_name == "gui_orm":
            from alpha_seed.utils.reward_score.vlm_verifiers.gui_orm_verifier import GUIORMVerifier
            result = GUIORMVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "instrruler":
            from alpha_seed.utils.reward_score.vlm_verifiers.instrruler_verifier import InstrRulerVerifier
            result = InstrRulerVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "text_function_call_v3":
            from alpha_seed.utils.reward_score.vlm_verifiers.text_function_call_v3_verifier import FunctionCallv3Verifier
            result = FunctionCallv3Verifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'visual_chained_tool_use_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.visual_chained_tool_use_verifier import VisualChainedToolUseVerifier
            result = VisualChainedToolUseVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'seed_grm_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.seed_grm_verifier import SeedGRMVerifier
            result = SeedGRMVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == "text_grm_instruction_follow_verifier":
            from alpha_seed.utils.reward_score.vlm_verifiers.text_grm_instruction_follow_verifier import TextGRMInstructionFollowVerifierVolc
            result = TextGRMInstructionFollowVerifierVolc(
                volc_ark_key=volc_ark_key, volc_model_name=volc_model_name).verify(response=response,
                                                                                   verifier_feature_dict=feature)
        elif verifier_name == 'mixed_perception_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.mixed_perception_verifier import ModelBasedMixedPerceptionVerifierVolc
            result = ModelBasedMixedPerceptionVerifierVolc(
                volc_ark_key=volc_ark_key, volc_model_name=volc_model_name).verify(response=response,
                                                                                   verifier_feature_dict=feature)
        elif verifier_name == 'grm_maintask_verifier':
            from alpha_seed.utils.reward_score.vlm_verifiers.grm_label_verifier import RuleBasedMaintaskVerifier
            result = RuleBasedMaintaskVerifier().verify(response=response, verifier_feature_dict=feature)
        elif verifier_name == 'video_cqa':
            from alpha_seed.utils.reward_score.vlm_verifiers.mcqa_verifier import VideoMCQAVerifier
            result = VideoMCQAVerifier().verify(response=response, verifier_feature_dict=feature)
        else:
            raise NotImplementedError(f'No verifier named "{verifier_name}".')

        final_score = (result.score * 2) - 1  # Convert [0, 1] (SeedRL) to [-1, +1] (AlphaSeed).
        status = json.dumps(
            {
                'tag': 'verified',
                'pred': result.extracted_answer,
                'answer': answer,
                'verifier_name': verifier_name,
                'score': final_score,
                'no_thinking_required': bool(no_thinking_required),
            },
            ensure_ascii=False)
        print(f'[VLM VERIFIER INFO] {status}')
        return final_score

    except ExtractAnswerFailed as e:
        status = json.dumps(
            {
                'tag': f'parsing_fail: {e}',
                'pred': full_rollout,
                'answer': answer,
                'verifier_name': verifier_name,
                # Assign lower scores to ill-formatted answers compared to incorrect but well-formatted answers.
                'score': -1.2,
                'no_thinking_required': bool(no_thinking_required),
            },
            ensure_ascii=False)
        print(f'[VLM VERIFIER WARNING] {status}')
        return -1.2

    except VerifierFailed as e:
        status = json.dumps(
            {
                'tag': f'verifier service failed: {e}',
                'pred': full_rollout,
                'verifier_feature': feature,
                'score': -2.0,
                'no_thinking_required': bool(no_thinking_required),
            },
            ensure_ascii=False)
        print(f'[VLM VERIFIER ERROR] {status}')
        return -2

    except Exception:
        import sys
        import traceback
        tb = ''.join(traceback.format_exception(*sys.exc_info()))  # noqa
        status = json.dumps(
            {
                'tag': f'verification error: {tb}',
                'pred': full_rollout,
                'verifier_feature': feature,
                'score': -2.0,
                'no_thinking_required': bool(no_thinking_required),
            },
            ensure_ascii=False)
        print(f'[VLM VERIFIER ERROR] {status}')
        return -2
