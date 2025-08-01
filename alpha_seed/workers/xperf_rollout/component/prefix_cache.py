import collections
import logging
import torch
import triton
import triton.language as tl
from typing import Optional, List

from xperf_gpt.utils import logging_rank_only


@triton.jit
def load_or_store_prefix_cache_kernel(kv_cache_ptr_array, prefix_cache_ptr, page_table_ptr, kv_slot_begin, kv_slot_end,
                                      slot_block_size: tl.constexpr, head_num: tl.constexpr, head_dim: tl.constexpr,
                                      max_cache_length: tl.constexpr, copy_block_dim_padded: tl.constexpr,
                                      is_load: tl.constexpr):
    # kv_cache: max_slot * slot_block_size * head_num * 2 * head_dim
    # prefix_cache: layer_num * max_cache_length * head_num * 2 * head_dim
    #
    # grid: [layer_num, slot_block_size]
    # block: [head_num * 2 * head_dim]

    layer_id = tl.program_id(0)
    pos_in_slot_block = tl.program_id(1)

    dtype = prefix_cache_ptr.dtype.element_ty
    kv_cache_ptr = tl.load(kv_cache_ptr_array + layer_id).to(tl.pointer_type(dtype))
    copy_block_dim = head_num * 2 * head_dim

    for i in range(kv_slot_begin, kv_slot_end):
        slot_id = tl.load(page_table_ptr + i)

        kv_cache_base_ptr = kv_cache_ptr + \
                            (slot_id * slot_block_size + pos_in_slot_block) * copy_block_dim
        prefix_cache_base_ptr = prefix_cache_ptr + \
                                (layer_id * max_cache_length + i * slot_block_size + pos_in_slot_block) * copy_block_dim

        offsets = tl.arange(0, copy_block_dim_padded)
        masks = tl.arange(0, copy_block_dim_padded) < copy_block_dim
        if is_load:
            loaded_val = tl.load(prefix_cache_base_ptr + offsets, mask=masks)
            tl.store(kv_cache_base_ptr + offsets, loaded_val)
        else:
            loaded_val = tl.load(kv_cache_base_ptr + offsets, mask=masks)
            tl.store(prefix_cache_base_ptr + offsets, loaded_val)


def load_or_store_prefix_cache(kv_cache_per_layer: List[torch.Tensor], prefix_cache, slot_id, kv_slot_begin,
                               kv_slot_end, page_table, is_load: bool):
    prefix_cache = prefix_cache[slot_id]
    layer_num, max_cache_length, head_num, _, head_dim = prefix_cache.shape
    assert len(kv_cache_per_layer) == layer_num
    slot_block_size = kv_cache_per_layer[0].shape[1]
    kv_cache_ptrs_array = torch.tensor([kv_cache.data_ptr() for kv_cache in kv_cache_per_layer],
                                       dtype=torch.long,
                                       device="cuda")

    grid = (layer_num, slot_block_size)
    copy_block_dim_padded = triton.next_power_of_2(head_num * 2 * head_dim)
    load_or_store_prefix_cache_kernel[grid](kv_cache_ptrs_array, prefix_cache, page_table, kv_slot_begin, kv_slot_end,
                                            slot_block_size, head_num, head_dim, max_cache_length,
                                            copy_block_dim_padded, is_load)


class PrefixCache(object):

    def __init__(self, kv_head_num, head_dim, layer_num, kv_cache_dtype, max_cache_slot_num, max_cache_length,
                 enable_paged_attn, slot_block_size):
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.kv_cache_dtype = kv_cache_dtype
        self.max_cache_slot_num = max_cache_slot_num
        self.max_cache_length = max_cache_length
        self.enable_paged_attn = enable_paged_attn
        self.slot_block_size = slot_block_size

        self.free_slots = set(range(self.max_cache_slot_num))
        self.cached_input_ids = torch.empty([self.max_cache_slot_num, self.max_cache_length],
                                            dtype=torch.long,
                                            device="cuda")
        self.cached_input_id_length = [0 for _ in range(self.max_cache_slot_num)]

        self.cache_slots = torch.empty(
            [self.max_cache_slot_num, self.layer_num, self.max_cache_length, self.kv_head_num, 2, self.head_dim],
            dtype=self.kv_cache_dtype,
            pin_memory=True)

        # slot_id -> req_id
        self.lru_cache = collections.OrderedDict()
        self.req_id_to_slot_id = {}

        logging_rank_only(logging.warning, 0, f"enable prefix cache with max cache slot num: {self.max_cache_slot_num}")

    def clear_cache(self):
        self.free_slots = set(range(self.max_cache_slot_num))
        self.cached_input_id_length = [0 for _ in range(self.max_cache_slot_num)]
        self.lru_cache.clear()
        self.req_id_to_slot_id.clear()

    def _get_free_slot(self, request_id: str):
        if len(self.free_slots) > 0:
            free_slot_id = self.free_slots.pop()
            self.lru_cache[free_slot_id] = request_id
            self.req_id_to_slot_id[request_id] = free_slot_id
            logging_rank_only(logging.warning, 0,
                              f"allocate cache slot, request_id: {request_id}, slot_id: {free_slot_id}")
            return free_slot_id, False
        else:
            # evict an existing slot
            slot_evict, request_id_evict = self.lru_cache.popitem(last=False)
            self.lru_cache[slot_evict] = request_id
            del self.req_id_to_slot_id[request_id_evict]
            self.req_id_to_slot_id[request_id] = slot_evict
            self.cached_input_id_length[slot_evict] = 0
            logging_rank_only(
                logging.warning, 0, f"evict cache slot, request_id: {request_id}, slot_id: {slot_evict}, "
                f"evicted request_id: {request_id_evict}")
            return slot_evict, True

    def calc_prefix_length(self, request_id: str, full_input_ids: torch.Tensor):
        if request_id not in self.req_id_to_slot_id:
            return 0
        slot_id = self.req_id_to_slot_id[request_id]
        cached_input_ids = self.cached_input_ids[slot_id]
        cached_input_id_length = self.cached_input_id_length[slot_id]
        min_length = min(cached_input_id_length, full_input_ids.shape[0])
        non_zero_pos = torch.nonzero(cached_input_ids[:min_length] - full_input_ids[:min_length])
        if non_zero_pos.numel() == 0:
            prefix_length = min_length
        else:
            prefix_length = non_zero_pos.flatten()[0].item()
        return prefix_length

    def load_from_cache(self, request_id: str, input_ids: torch.Tensor, kv_cache_table: torch.Tensor, xperf_module,
                        start_pos: Optional[int]):
        if request_id not in self.req_id_to_slot_id:
            return 0
        else:
            slot_id = self.req_id_to_slot_id[request_id]
            del self.lru_cache[slot_id]
            self.lru_cache[slot_id] = request_id

        if start_pos is None:
            prefix_length = self.calc_prefix_length(request_id, input_ids)
        else:
            prefix_length = start_pos
        if prefix_length == 0:
            return 0

        logging_rank_only(logging.warning, 0,
                          f"hit prefix cache, request_id: {request_id}, prefix length: {prefix_length}")
        is_cache_quant = self.kv_cache_dtype not in [torch.bfloat16, torch.float16]
        if self.enable_paged_attn:
            # slot * slot_block_size * kv_head * 2 * head_dim
            kv_slot_count = (prefix_length + self.slot_block_size - 1) // self.slot_block_size
            # for layer_id in range(self.layer_num):
            #     kv_cache = xperf_module._get_kv_cache(layer_id, is_cache_quant)
            #     for kv_slot in range(kv_slot_count):
            #         cached = self.cache_slots[
            #                     slot_id,
            #                     layer_id,
            #                     kv_slot * self.slot_block_size: (kv_slot + 1) * self.slot_block_size
            #                  ]
            #         kv_slot_id = kv_cache_table[kv_slot].item()
            #         kv_cache[kv_slot_id] = cached.cuda()
            kv_cache_per_layer = [
                xperf_module._get_kv_cache(layer_id, is_cache_quant) for layer_id in range(self.layer_num)
            ]
            load_or_store_prefix_cache(kv_cache_per_layer, self.cache_slots, slot_id, 0, kv_slot_count, kv_cache_table,
                                       True)
        else:
            # 2HBSD
            kv_slot_id = kv_cache_table.item()
            for layer_id in range(self.layer_num):
                kv_cache = xperf_module._get_kv_cache(layer_id, is_cache_quant)
                # length * head_num * 2 * head_dim
                cached = self.cache_slots[slot_id, layer_id, :prefix_length]
                kv_cache[:, :, kv_slot_id, :prefix_length] = cached.permute(2, 1, 0, 3)

        return prefix_length

    def save_to_cache(self, request_id: str, full_input_ids: torch.Tensor, kv_cache_table: torch.Tensor, xperf_module):
        if request_id not in self.req_id_to_slot_id:
            slot_id, is_evict = self._get_free_slot(request_id)
        else:
            slot_id = self.req_id_to_slot_id[request_id]
            is_evict = False

        prefix_length = self.calc_prefix_length(request_id, full_input_ids)

        total_length = min(full_input_ids.shape[0], self.max_cache_length)
        if prefix_length == total_length:
            return
        self.cached_input_ids[slot_id, prefix_length:total_length] = full_input_ids[prefix_length:total_length]
        self.cached_input_id_length[slot_id] = total_length
        logging_rank_only(
            logging.warning, 0,
            f"save prefix cache, request_id: {request_id}, prefix length: {prefix_length}, total length: {total_length}"
        )
        is_cache_quant = self.kv_cache_dtype not in [torch.bfloat16, torch.float16]
        # TODO: use triton kernel to impl
        if self.enable_paged_attn:
            # slot * slot_block_size * kv_head * 2 * head_dim
            kv_slot_start = prefix_length // self.slot_block_size
            kv_slot_end = (total_length + self.slot_block_size - 1) // self.slot_block_size
            #for layer_id in range(self.layer_num):
            #    kv_cache = xperf_module._get_kv_cache(layer_id, is_cache_quant)
            #    for kv_slot in range(kv_slot_start, kv_slot_end):
            #        kv_slot_id = kv_cache_table[kv_slot].item()
            #        self.cache_slots[
            #            slot_id,
            #            layer_id,
            #            kv_slot * self.slot_block_size: (kv_slot + 1) * self.slot_block_size
            #        ] = kv_cache[kv_slot_id].cpu()
            kv_cache_per_layer = [
                xperf_module._get_kv_cache(layer_id, is_cache_quant) for layer_id in range(self.layer_num)
            ]
            load_or_store_prefix_cache(kv_cache_per_layer, self.cache_slots, slot_id, kv_slot_start, kv_slot_end,
                                       kv_cache_table, False)
        else:
            # 2HBSD
            kv_slot_id = kv_cache_table.item()
            for layer_id in range(self.layer_num):
                kv_cache = xperf_module._get_kv_cache(layer_id, is_cache_quant)
                self.cache_slots[slot_id, layer_id, prefix_length: total_length] = \
                    kv_cache[:, :, kv_slot_id, prefix_length: total_length].permute(2, 1, 0, 3)

        return is_evict