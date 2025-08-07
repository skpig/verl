from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.tools import get_valid_options, extract_option


class MCQAVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        options = list(get_valid_options(verifier_feature_dict.get('options', None)))
        if len(answer) == 0 and answer in options:
            gt_option = answer
        else:
            try:
                gt_option = extract_option(answer, options)
                assert gt_option
            except:
                raise ExtractAnswerFailed(f"Failed to extract ground truth option from: {answer}")

        try:
            pred_option = extract_option(response, options)
            assert pred_option
        except:
            raise ExtractAnswerFailed(f"Failed to extract predict option from: {response}")

        score = float(gt_option == pred_option)
        return VerifyResult(score=score, extracted_answer=pred_option)


if __name__ == "__main__":
    import pandas as pd
    import json

    input_file = "/mnt/bn/ic-vlm/wangjw/dataset/seed_rl_train/HAIC_train_thinktag_15B-v57-cotv4_0321/1.parquet"
    data = pd.read_parquet(input_file).to_dict("records")

    gui_verifier = MCQAVerifier()
    for i, d in enumerate(data):
        verifier_feature = json.loads(d['session']['verifier_feature'])
        gt = verifier_feature['answer']
        pred = verifier_feature['gt_response']
        res = gui_verifier.verify(pred, verifier_feature_dict=verifier_feature)
        status = json.dumps({
            'tag': 'verified',
            'pred': res.extracted_answer,
            'answer': gt,
            'score': res.score
        },
                            ensure_ascii=False)
        if res.score == 0:
            print(f"----{i}----")
            print(f'[VERIFIER INFO] {status}', flush=True)
