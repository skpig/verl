import re
import json
import importlib

from alpha_seed.utils.reward_score.vlm_verifiers.augment_rules_if import process_results_ifeval
from alpha_seed.utils.reward_score.vlm_verifiers.augment_rules_cl import CONSTRAIN_MANAGER

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult, VerifierFailed


def rule_checker_zh(response, data_source):
    if response == "" or len(response) == 0:
        return 0

    module = importlib.import_module("alpha_seed.utils.reward_score.vlm_verifiers.augment_rules_zh")
    json_dict = json.loads(data_source.split("instrruler<RULESPLITTOKEN>")[1][2:])
    y_clause = []
    for rule, slot in zip(json_dict["rules"], json_dict["slots"]):
        obj = getattr(module, f"Rule_{rule}", None)()
        try:
            result = obj.check(response, slot)
        except:
            result = False
        y_clause.append(result)

    x = []
    for logic in json_dict["verification"]:
        result = eval(logic)
        x.append(result)
    #if "baseline" not in json_dict:
    #    json_dict["baseline"] = 1
    #final_result = eval(json_dict['verification'])
    final_result = float(all(x))

    tag_string = json_dict["tag"]
    # print("<><><><><><><><><><> >>>>>>>>>>>>>>>>>>>>")
    # print("response: ", response)
    # print("json_dict:", json_dict)
    # print("x       : ", x)
    # print("final_result: ", final_result)
    # print("<><><><><><><><><><> <<<<<<<<<<<<<<<<<<<<")
    return final_result


def rule_checker_en(response, data_source):
    if response == "" or len(response) == 0:
        return 0

    module = importlib.import_module("alpha_seed.utils.reward_score.vlm_verifiers.augment_rules_en")
    json_dict = json.loads(data_source.split("instrruler<RULESPLITTOKEN>")[1][2:])
    y_clause = []
    for rule, slot in zip(json_dict["rules"], json_dict["slots"]):
        obj = getattr(module, f"Rule_{rule}", None)()
        try:
            result = obj.check(response, slot)
        except:
            result = False
        y_clause.append(result)

    x = []
    for logic in json_dict["verification"]:
        result = eval(logic)
        x.append(result)
    #if "baseline" not in json_dict:
    #    json_dict["baseline"] = 1
    #final_result = eval(json_dict['verification'])
    final_result = float(all(x))

    tag_string = json_dict["tag"]
    # print("<><><><><><><><><><> >>>>>>>>>>>>>>>>>>>>")
    # print("response: ", response)
    # print("json_dict:", json_dict)
    # print("x       : ", x)
    # print("final_result: ", final_result)
    # print("<><><><><><><><><><> <<<<<<<<<<<<<<<<<<<<")
    return final_result


def rule_checker_cl(response, data_source):
    if response == "" or len(response) == 0:
        return 0

    constrain_str = data_source.split("instrruler<RULESPLITTOKEN>")[1][2:]
    try:
        decoded_constrain = CONSTRAIN_MANAGER.load_constrain(constrain_str)
    except Exception as e:
        print(constrain_str)
        decoded_constrain = None

    try:
        all_cons_valid = CONSTRAIN_MANAGER.gerneral_check(decoded_constrain, response)
        final_result = float(all_cons_valid)
    except Exception as e:
        final_result = 0.0

    #if "baseline" not in json_dict:
    #    json_dict["baseline"] = 1
    #final_result = eval(json_dict['verification'])
    #final_result = float(all(x))

    tag_string = "collie"
    # print("<><><><><><><><><><> >>>>>>>>>>>>>>>>>>>>")
    # print("response: ", response)
    # print("final_result: ", final_result)
    # print("<><><><><><><><><><> <<<<<<<<<<<<<<<<<<<<")
    return final_result


def rule_checker_if(response, data_source):
    if response == "" or len(response) == 0:
        return 0

    try:
        verification_json = json.loads(data_source.split("instrruler<RULESPLITTOKEN>")[1][2:])
        verification_json["prompt"] = ""
        results = [response]
        checking = process_results_ifeval(verification_json, results)
        y_clause = checking["inst_level_loose_acc"]
    except:
        y_clause = [False]

    x = y_clause
    #if "baseline" not in json_dict:
    #    json_dict["baseline"] = 1
    #final_result = eval(json_dict['verification'])
    final_result = float(all(x))

    tag_string = "ifeval"
    # print("<><><><><><><><><><> >>>>>>>>>>>>>>>>>>>>")
    # print("response: ", response)
    # print("x       : ", x)
    # print("final_result: ", final_result)
    # print("<><><><><><><><><><> <<<<<<<<<<<<<<<<<<<<")
    return final_result


class InstrRulerVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        data_source = verifier_feature_dict['constraint']
        if response == "":
            raise ExtractAnswerFailed
        if data_source.startswith("instrruler<RULESPLITTOKEN>ZH"):
            ruler_checker_score = rule_checker_zh(response, data_source)
            return VerifyResult(score=ruler_checker_score, extracted_answer=response)
        elif data_source.startswith("instrruler<RULESPLITTOKEN>EN"):
            ruler_checker_score = rule_checker_en(response, data_source)
            return VerifyResult(score=ruler_checker_score, extracted_answer=response)
        elif data_source.startswith("instrruler<RULESPLITTOKEN>CL"):
            ruler_checker_score = rule_checker_cl(response, data_source)
            return VerifyResult(score=ruler_checker_score, extracted_answer=response)
        elif data_source.startswith("instrruler<RULESPLITTOKEN>IF"):
            ruler_checker_score = rule_checker_if(response, data_source)
            return VerifyResult(score=ruler_checker_score, extracted_answer=response)
        return VerifierFailed(message=f"[Instrruler] Unknown data source {data_source}")
