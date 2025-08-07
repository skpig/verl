import re

from word2number import w2n

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.tools import remove_bold


def text_normalization(text):
    text = remove_bold(text).lower()
    text = re.sub(r'\s+', ' ', text)
    words = []
    for word in text.split(" "):
        if word in w2n.american_number_system:
            word = str(w2n.word_to_num(word))
        words.append(word)
    return ' '.join(words).strip()


def extract_number(text):
    text = text_normalization(text)
    answer = re.findall(r'^(\d+)', text)
    if len(answer):
        return int(answer[0])
    answer = re.findall(r'\\boxed\{(\d+)\}', text)
    if len(answer):
        return int(answer[0])

    # has English prefix
    answer = re.findall(r"answer(\ is|:|\ is:)\ ?(\d+)", text, re.DOTALL)
    if len(answer):
        return int(answer[0][-1])

    # has Chinese prefix
    answer = re.findall(r"(回答|答案)(是|：|是：)\ ?(\d+)", text, re.DOTALL)
    if len(answer):
        return int(answer[0][-1])

    # finally, try to get the first number
    answer = re.findall(r"\d+", answer)
    if len(answer):
        return int(answer[0])
    return


class ActionCountVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict, delta: float) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        if isinstance(answer, int):
            gt_number = answer
        else:
            answer = str(answer)
            try:
                gt_number = extract_number(answer)
                assert isinstance(gt_number, int)
            except:
                raise ExtractAnswerFailed(f"Failed to extract ground truth number from: {answer}")

        try:
            pred_number = extract_number(response)
            assert isinstance(pred_number, int)
        except:
            raise ExtractAnswerFailed(f"Failed to extract predict number from: {response}")

        if gt_number == pred_number:
            score = 1.0
        else:
            score = min(max(1 - abs(gt_number - pred_number) / gt_number, 0), delta) if gt_number > 0 else float(
                gt_number == pred_number)
        return VerifyResult(score=score, extracted_answer=pred_number)


if __name__ == "__main__":
    import pandas as pd
    import json

    input_file = "/mnt/bn/ic-vlm/wangjw/dataset/seed_rl_train/RepCount_trian_val_thinktag_15B-v57-cotv4_0321/1.parquet"
    data = pd.read_parquet(input_file).to_dict("records")

    gui_verifier = ActionCountVerifier()
    for i, d in enumerate(data):
        verifier_feature = json.loads(d['session']['verifier_feature'])
        gt = verifier_feature['answer']
        pred = verifier_feature['gt_response']
        res = gui_verifier.verify(pred, verifier_feature_dict=verifier_feature, delta=0.8)
        status = json.dumps({
            'tag': 'verified',
            'pred': res.extracted_answer,
            'answer': gt,
            'score': res.score
        },
                            ensure_ascii=False)
        if res.score < 1:
            print(f"----{i}----")
            print(f'[VERIFIER INFO] {status}', flush=True)
