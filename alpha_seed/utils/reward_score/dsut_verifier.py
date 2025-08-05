from typing import *
import json
import dill
from alpha_seed.utils.reward_score import _select_rm_score_fn
from alpha_seed.workers.xperf_rollout.component.query_plugin import EnvStates


def compute_score(solution_str, ground_truth, **kwargs):
    ground_truth = json.loads(ground_truth)
    if ground_truth['verifiable_meta']['data_type'] == 'json':
        match_end = -1
        for match_end in range(len(solution_str) - 1, -1, -1):
            if solution_str[match_end] == '}':
                break
        if match_end > 0:
            brackets = 1
            for match_start in range(match_end - 1, -1, -1):
                if solution_str[match_start] == '{':
                    brackets -= 1
                    if brackets == 0:
                        break
                elif solution_str[match_start] == '}':
                    brackets += 1
            if match_start >= 0:
                json_response = solution_str[match_start:match_end + 1]
                # print('JSON response:', json_response)
                try:
                    # If the JSON object is not standard (e.g., it contains None, False, True)
                    # json.loads will fail
                    json_response = json.loads(json_response)
                except:
                    # Should we consider safety check here?
                    try:
                        json_response = eval(json_response)
                    except Exception as e:
                        # print('Can not parse the JSON response:', e)
                        return -1.0
                json_ground_truth = ground_truth['verifiable_answer']

                # NOTE: These dumping and loading operations remove all spaces in the JSON object,
                # because spaces are sometimes confusing and are not important in this task.
                # We convert the string to JSON object again to disregard the order of the keys.
                # Note that sometimes the extracted part can not be dumped because it contains sets.
                try:
                    cleaned_json_response = json.loads(json.dumps(json_response).replace(' ', ''))
                except Exception as e:
                    print('Can not parse the extracted JSON response:', str(json_response), e)
                    return -1.0
                cleaned_json_ground_truth = json.loads(json.dumps(json_ground_truth).replace(' ', ''))
                if cleaned_json_response == cleaned_json_ground_truth:
                    return 1.0
                else:
                    return -1.0

        # print("No JSON is found in the response", response_text)
        return -1.0
    else:
        print('DEBUG', ground_truth['verifiable_meta']['data_type'])
        raise NotImplementedError
