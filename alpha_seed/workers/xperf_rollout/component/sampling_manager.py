# adapted from orca_server directly

import torch
from typing import Dict


def _sample(probs: torch.Tensor):
    noise = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / noise, dim=-1).reshape(-1, 1)


class Sampler:

    def __init__(self, generation_config: Dict) -> None:
        torch.manual_seed(9898)
        self.metrics = None
        self.return_log_probs = True
        try:
            # TODO: adhoc impl, does no type check
            self.top_k = generation_config["top_k"]
            self.top_p = generation_config["top_p"]
            self.do_sample = generation_config["do_sample"]
            self.temperature = generation_config["temperature"]
            self.logits_manipulate_fn = generation_config.get("logits_manipulate_fn", None)
        except Exception as e:
            raise ValueError(f"required key missing from generation_config: {str(e)}")

        if self.do_sample:
            assert self.top_p or self.top_k, f"invalid top_p {self.top_p}, top_k {self.top_k}"

    # ================== Copied & Modified from xperf_gpt/lego ==================
    # TODO: shapes is yet to be confirmed
    def sample(self,
               logits: torch.Tensor,
               need_torch_tensor: bool = False,
               history_ids=None,
               sample_kwargs={}) -> torch.tensor:
        """
        logits of shape [bs, vocab_size]
        """
        # import ipdb; ipdb.set_trace()
        next_token_scores = logits.contiguous()
        if self.logits_manipulate_fn is not None:
            next_token_scores = self.logits_manipulate_fn(next_token_scores, history_ids)
        if self.do_sample:
            next_token_scores = self.temperature_logits_wrapper(scores=next_token_scores,
                                                                per_query_arg=sample_kwargs.get("temperature", [None]))
        if not self.do_sample:
            tokens = torch.argmax(next_token_scores, dim=-1).cpu().tolist() if not need_torch_tensor else torch.argmax(
                next_token_scores, dim=-1)
            return tokens, None, None, None
        if self.do_sample:
            next_token_scores = self.topk_logits_wrapper(scores=next_token_scores,
                                                         per_query_arg=sample_kwargs.get("top_k", [None]))
            next_token_scores = self.topp_logits_wrapper(scores=next_token_scores,
                                                         per_query_arg=sample_kwargs.get("top_p", [None]))

        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        tokens = _sample(probs).cpu().squeeze(1).tolist() if not need_torch_tensor else _sample(probs)
        token_probs = torch.gather(probs, 1, tokens)
        log_probs = None
        if self.metrics is not None:
            self.metrics["sample_token_num"] += token_probs.numel()
            self.metrics['prob_mean'] += token_probs.sum().item()
            self.metrics['prob_lt_0.0001'] += (token_probs < 0.0001).sum().item()
            self.metrics['prob_lt_1e-5'] += (token_probs < 1e-5).sum().item()
            self.metrics['prob_lt_1e-6'] += (token_probs < 1e-6).sum().item()
        if self.return_log_probs:
            log_probs = torch.log(token_probs)
        probs_gt_threshold_num = torch.gt(probs, 1e-4).float().sum(-1)
        probs_lt_threshold_sum = (torch.lt(probs, 1e-4).float() * probs).sum(-1)
        return tokens.view(-1), log_probs.view(-1), probs_gt_threshold_num.view(-1), probs_lt_threshold_sum.view(-1)

    def set_generator_strategy(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            if key not in self.__dict__.keys():
                continue
            elif self.__dict__[key] is not None and type(self.__dict__[key])(value) != value:
                raise ValueError(
                    "the argument '{}' should be '{}' but got '{}' and cannot be converted implicitly".format(
                        key,
                        type(self.__dict__[key]).__name__,
                        type(value).__name__))
            else:
                self.__dict__[key] = type(self.__dict__[key])(value) if self.__dict__[key] is not None else value

    def per_query_process(func):

        def wrap(self, *args, **kwargs):
            scores = kwargs.pop("scores")
            per_query_arg = kwargs.pop("per_query_arg")
            use_batch = all(element is None for element in per_query_arg)
            if use_batch:
                return func(self, *args, scores=scores, **kwargs)
            else:
                output_scores = []
                for i in range(len(per_query_arg)):
                    output_scores.append(
                        func(self, *args, scores=scores[i:i + 1], per_query_arg=per_query_arg[i], **kwargs))
                return torch.cat(output_scores, dim=0)

        return wrap

    @per_query_process
    def temperature_logits_wrapper(self, scores: torch.Tensor, per_query_arg=None):
        temperature = per_query_arg or self.temperature
        if temperature != 1.0:
            scores = scores / temperature
        return scores

    @per_query_process
    def topk_logits_wrapper(self, scores: torch.FloatTensor, filter_value: float = -float("Inf"), per_query_arg=None):
        top_k = per_query_arg or self.top_k
        if top_k > 0:
            scores = torch.classes.XGPT.TopkTopp().topk_sample(scores, top_k, filter_value)
        return scores

    @per_query_process
    def topp_logits_wrapper(self,
                            scores: torch.FloatTensor,
                            filter_value: float = -float("Inf"),
                            min_tokens_to_keep: int = 1,
                            per_query_arg=None):
        top_p = per_query_arg or self.top_p
        if top_p > 0.0 and top_p < 1.0:
            score_sorted, idx_sorted = torch.sort(scores, dim=-1, descending=True)
            scores = torch.classes.XGPT.TopkTopp().topp_sample(
                scores,
                score_sorted,
                idx_sorted,
                top_p,
                filter_value,
                min_tokens_to_keep,
            )
        return scores

    def topk_logits_wrapper_vanilla(self, scores: torch.FloatTensor, filter_value: float = -float("Inf")):
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, filter_value)
        return scores

    def topp_logits_wrapper_vanilla(self,
                                    scores: torch.FloatTensor,
                                    filter_value: float = -float("Inf"),
                                    min_tokens_to_keep: int = 1):
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        sorted_logits = sorted_logits.softmax(dim=-1)
        cumulative_probs = [sorted_logits[i].cumsum(dim=-1) for i in range(sorted_logits.shape[0])]
        cumulative_probs = torch.stack(cumulative_probs)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, filter_value)
        return scores
