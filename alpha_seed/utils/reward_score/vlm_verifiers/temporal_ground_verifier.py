import re

import numpy as np
import torch

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.tools import remove_bold


def temporal_area(windows):
    """
    Compute the areas of temporal windows.

    Args:
        windows (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed. They
            are expected to be in ``(start, end)`` format.

    Returns:
        :obj:`nn.Tensor[N]`: The computed areas.
    """
    return windows[:, 1] - windows[:, 0]


def temporal_intersection(windows1, windows2, aligned=False):
    """
    Compute the intersections among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the intersections
            among aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed \
            intersection values.
    """
    if aligned:
        s = torch.max(windows1[:, 0], windows2[:, 0])
        e = torch.min(windows1[:, 1], windows2[:, 1])
    else:
        s = torch.max(windows1[:, None, 0], windows2[:, 0])
        e = torch.min(windows1[:, None, 1], windows2[:, 1])

    inter = (e - s).clamp(0)
    return inter


def temporal_iou(windows1, windows2, aligned=False):
    """
    Compute the intersection-over-unions (IoUs) among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the IoU among
            aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed pairwise \
            IoU values.
    """
    area1 = temporal_area(windows1)
    area2 = temporal_area(windows2)

    inter = temporal_intersection(windows1, windows2, aligned=aligned)

    if aligned:
        iou = inter / (area1 + area2 - inter)
    else:
        iou = inter / (area1[:, None] + area2 - inter)

    return iou


def text_normalization(text):
    text = remove_bold(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_time_segments(ans):
    ans = ans.lower()
    sentences = re.split(r'[!?\n]', ans)

    keywords = ['starts', 'ends', 'happens in', 'start time', 'end time', 'start', 'end', 'happen']
    candidates = []
    for sentence in sentences:
        if any(keyword in sentence for keyword in keywords):
            candidates.append(sentence)

    timestamps = []
    patterns = [r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"]

    for time_pattern in patterns:
        time_matches = re.findall(time_pattern, ans)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]

    if len(timestamps) == 0:
        pattern = r"(\d+\.*\d*)\s* to \s*(\d+\.*\d*)"
        time_matches = re.findall(pattern, ans)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]

    if len(sentences) == 0:
        return None

    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r'\b(\d+\.\d+\b|\b\d+)\b')
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                time_in_sec = float(time[0])
                times.append(time_in_sec)
        times = times[:len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r'\b((\d{1,2}:\d{2}:\d{2}))\b')
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                t = time[0]
            else:
                continue
            if t.count(':') == 2:
                h, m, s = map(int, t.split(':'))
                time_in_sec = h * 3600 + m * 60 + s
            elif t.count(':') == 1:
                m, s = map(int, t.split(':'))
                time_in_sec = m * 60 + s
            times.append(time_in_sec)
        times = times[:len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    results = []
    for (start, end) in timestamps:
        if end > start:
            results.append([start, end])
        else:
            results.append([end, start])

    if len(results) == 0:
        results = None

    if results is not None:
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, list)
            assert len(item) == 2
            assert isinstance(item[0], (int, float))
            assert isinstance(item[1], (int, float))

    return results


def check_time_segments(segments):
    if not isinstance(segments, list) or len(segments) == 0:
        return None
    last_end = -1
    try:
        for seg in segments:
            assert seg[0] < seg[1] and seg[0] - last_end >= 1
            last_end = seg[1]
    except:
        return None
    return segments


def extract_answer(text):
    text = text_normalization(text)
    text = text.replace("[", "").replace("]", "")
    # print("pure_text:", text)

    # Find the answer in last response
    for pattern in [r"[a]nswer(\ is|:|\ is:)\ ?(.+)", r"(回答|答案)(是|：|是：)\ ?(.+)"]:
        answer = re.findall(pattern, text, re.DOTALL)
        if len(answer):
            text = answer[-1][-1]
            break

    segments = extract_time_segments(text)
    segments = check_time_segments(segments)
    # print(f"segments: {segments}")
    assert isinstance(segments, list) and len(segments)
    return segments


def calculate_iou(pred, gt):
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    f1_score = [0 for _ in iou_thr]
    rec_score = [0 for _ in iou_thr]
    prc_score = [0 for _ in iou_thr]

    gt = torch.Tensor(gt)
    pred = torch.Tensor(pred)
    iou = temporal_iou(gt, pred)

    for i, thr in enumerate(iou_thr):
        if iou.max() < thr:
            continue
        else:
            rec = (iou.amax(dim=1) >= thr).float().mean().item()
            prc = (iou.amax(dim=0) >= thr).float().mean().item()
            rec_score[i] = rec
            prc_score[i] = prc
            f1_score[i] = 2 * prc * rec / (prc + rec)

    out = dict()
    # for f1, thr in zip(f1_score, iou_thr):
    #     out[f'F1@{thr}'] = round(f1, 5)
    for i, thr in enumerate(iou_thr):
        out[f'F1@{thr}'] = round(f1_score[i], 3)
        out[f'R@{thr}'] = round(rec_score[i], 3)
        out[f'P@{thr}'] = round(prc_score[i], 3)
    out['miou'] = iou.max(dim=1)[0].mean().item() - min(0.5 * abs(len(pred) - len(gt)) / len(gt), 1)
    out['miou'] = max(0, out['miou'])
    return out


def calculate_score(pred, gt, score_type='avg_f1'):
    iou = calculate_iou(pred, gt)
    if score_type == 'avg_f1':
        score = iou['F1@0.7'] * 1 / 2 + iou['F1@0.5'] * 1 / 3 + iou['F1@0.3'] * 1 / 9 + iou['F1@0.1'] * 1 / 18
    elif score_type == 'miou':
        score = iou['miou']
    else:
        score = iou['F1@0.7'] * 1 / 2 + iou['F1@0.5'] * 1 / 3 + iou['F1@0.3'] * 1 / 9 + iou['F1@0.1'] * 1 / 18
    return score


class TemporalGroundVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        gt_segments = answer
        if isinstance(answer, np.ndarray):
            gt_segments = answer.tolist()
        elif isinstance(answer, str):
            try:
                gt_segments = eval(answer)
            except:
                gt_segments = extract_answer(answer)
        try:
            assert isinstance(gt_segments, list) and len(gt_segments)
        except:
            raise ExtractAnswerFailed(f"Failed to extract ground truth segments from: {answer}")

        try:
            pred_segments = extract_answer(response)
        except:
            raise ExtractAnswerFailed(f"Failed to extract predict segments from: {response}")

        score = calculate_score(pred_segments,
                                gt_segments,
                                score_type=verifier_feature_dict.get('score_type', 'avg_f1'))
        return VerifyResult(score=score, extracted_answer=pred_segments)


if __name__ == "__main__":
    import pandas as pd
    import json

    input_file = "/mnt/bn/ic-vlm/wangjw/dataset/seed_rl_train/ET-Instruct_tal_5k_thinktag_15B-v57-cotv4_0321/1.parquet"
    data = pd.read_parquet(input_file).to_dict("records")

    gui_verifier = TemporalGroundVerifier()
    for i, d in enumerate(data):
        verifier_feature = json.loads(d['session']['verifier_feature'])
        verifier_feature['score_type'] = 'miou'
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
        if res.score < 1:
            print(f"----{i}----")
            print(f'[VERIFIER INFO] {status}', flush=True)
