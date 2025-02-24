from typing import *
from dataclasses import dataclass
import uuid
import torch
import copy


@dataclass
class Query:
    id: Optional[uuid.UUID]
    idx: int
    original_input_ids: Optional[List[int]]
    input_ids: Optional[List[int]]
    code_book: Optional[List[int]]
    accepted_len: Optional[List[int]]
    input_prompt: Union[str, List[str]]
    input_len: Optional[int]
    new_token_ids: Optional[List[int]]
    new_token_log_probs: Optional[List[int]]
    kv_slot_ids: Optional[List[int]]
    is_context_computing: bool
    new_token_len: int
    context_shift: int
    output_prompt: Union[str, List[str]]
    prefix_already_computed_len: int
    multiround_id: int
    multiround_len: int
    system_ids_len: int
    first_scheduled_time: int
    first_token_time: int
    finished_time: int
    hidden_states: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    logits_mask: Optional[torch.Tensor]
    shift_label: Optional[torch.Tensor]
    cur_batch_pad_token: int
    nll_loss: Optional[torch.Tensor]
    is_finished: bool
    off_policy_steps: int
    meta_info: Optional[Dict] = None

    def __init__(self,
                 input_ids,
                 input_prompt,
                 idx,
                 prefix_already_computed_len=0,
                 system_ids_len=0,
                 code_book=None,
                 constraint_decoding_predictor=None):
        self.id = uuid.uuid4()
        self.idx = idx
        self.original_input_ids = copy.deepcopy(input_ids)
        self.input_ids = input_ids
        self.code_book = code_book
        self.accepted_len = []
        self.input_prompt = input_prompt
        self.prefix_already_computed_len = prefix_already_computed_len
        self.input_len = len(input_ids)
        self.is_context_computing = True
        self.new_token_ids = []
        self.new_token_log_probs = []
        self.kv_slot_ids = []
        self.new_token_len = 0
        self.output_prompt = ""
        self.context_shift = 0
        self.multiround_id = 0
        self.multiround_len = len(input_prompt) if isinstance(input_prompt, list) else 0
        self.multiround_input_len = 0
        self.multiround_new_token_len = 0
        self.system_ids_len = system_ids_len
        self.hidden_states = None
        self.logits = None
        self.logits_mask = None
        self.shift_label = None
        self.cur_batch_pad_token = 0
        self.nll_loss = None
        self.is_finished = False
        self.meta_info = None

        self.first_scheduled_time = 0
        self.first_token_time = 0
        self.finished_time = 0
        self.is_jumping = False
        self.jump_tokens = 0
        self.off_policy_steps = 0
        self.top_k = None
        self.top_p = None
        self.temperature = None
        self.max_new_tokens = None
        self.max_length = None

    # Check whether current query is going to enter the decoding stage
    def _is_to_decoding_compute(self):
        # already in decode stage
        if not self.is_context_computing:
            return True
        # called after context stage, no context_shift means it doesn't need context-split
        if self.is_context_computing and self.context_shift == 0:
            return True
        # for query which needs context-split, all input_ids finished context computing
        if self.is_context_computing and (self.context_shift + self.prefix_already_computed_len) == len(self.input_ids):
            return True
        return False

    def is_kv_cache_slot_allocated(self):
        return len(self.kv_slot_ids) > 0
