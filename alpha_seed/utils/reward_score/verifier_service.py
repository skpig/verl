import os
import sys
import json
import ray


def compute_score_client(solution_str, ground_truth, verifier_service_psm, data_uid, config, **argv) -> float:
    """Directly retrieve the scores from SandboxClient"""
    score = None
    if config.trainer.use_remote_verifier:
        # get the sandbox client endpoint
        handler = ray.get_actor('remote_client')
        # retrieve the score directly
        score = ray.get(handler.get_results.remote(data_uid))

    if score is None:
        score = compute_score(solution_str, ground_truth, verifier_service_psm, **argv)

    # optionally, compute the score with original code to compare the results
    # score_original = compute_score(solution_str, ground_truth, code_sandbox_psm, **argv)
    # assert score == score_original

    return score


# trainer.verifier_service_psm="'seed.alphaseed.verify_service?idc=yg&cluster=default'" \
def compute_score(solution_str, ground_truth, verifier_service_psm, **argv) -> float:
    import euler
    euler.install_thrift_import_hook()
    from alpha_seed.utils.reward_score.idls.verifier_thrift import VerifyService, VerifyRequest
    from alpha_seed.utils.reward_score.idls.base_thrift import Base
    client = euler.Client(VerifyService, f'sd://{verifier_service_psm}')
    for i in range(3):
        try:
            if solution_str.startswith("A conversation between user and assistant."):
                solution_str = solution_str[400:]
            if isinstance(ground_truth, str):
                ground_truth = json.loads(ground_truth)
            problem = ground_truth["problem"]
            reference_answer = ground_truth["reference_answer"]
            if isinstance(reference_answer, int):
                reference_answer = str(reference_answer)
            verify_type = ground_truth["verify_type"]
            req = VerifyRequest(problem=problem,
                                reference_answer=reference_answer,
                                generated_response=solution_str,
                                verify_type=verify_type,
                                base_p=Base(
                                    Extra={
                                        "arnold_trial_id": os.environ.get("ARNOLD_TRIAL_ID", "0"),
                                        "arnold_trial_owner": os.environ.get("ARNOLD_TRIAL_OWNER", "0")
                                    }))
            resp = client.verify(req, timeout=120)
            if resp.is_correct == True:
                return 1
            else:
                return -1
        except Exception as ex:
            continue
    print(f'Got exception in compute_score via verifier_service:')
    return -1
