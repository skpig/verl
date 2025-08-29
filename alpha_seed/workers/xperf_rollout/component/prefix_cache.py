import collections
import logging
import math
import torch
import triton
import triton.language as tl
from dataclasses import dataclass, field
from typing import Optional, List

from xperf_gpt.utils import logging_rank_only

_alloc_pinned_module = None


def allocate_pinned_memory(shape, dtype):
    cuda_src = """
    #include <torch/extension.h>
    #include <ATen/ATen.h>
    #include <cuda_runtime.h>
    #include <stdexcept>
    #include <memory>

    namespace {

    struct HostPtr {
      void* p{nullptr};
      size_t nbytes{0};
      ~HostPtr() {
        if (p) cudaFreeHost(p);  // 与 cudaHostAlloc 配套释放
      }
    };

    void checkCuda(cudaError_t e) {
      if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
      }
    }

    } // anonymous

    torch::Tensor alloc_pinned(int64_t nbytes) {
      // 申请 pinned 内存
      void* ptr = nullptr;
      checkCuda(cudaHostAlloc(&ptr, nbytes, cudaHostAllocPortable)); // 可选 flags

      // 用智能指针托管，确保异常时也能释放
      auto hptr = std::make_shared<HostPtr>();
      hptr->p = ptr;
      hptr->nbytes = nbytes;

      // 用 from_blob 绑定，并标记 pinned_memory(true)
      auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU).pinned_memory(true);

      // 自定义 deleter：当最后一个张量引用释放后，自动 cudaFreeHost
      auto deleter = [hptr](void* p) mutable {
        // 让 hptr 析构时调用 cudaFreeHost
        hptr.reset();
      };

      // strides 留空则按行优先
      auto t = torch::from_blob(ptr, torch::IntArrayRef({nbytes}), deleter, options);

      // 为了安全，返回一个拥有自己 storage 的张量拷贝视图（不复制数据，只拷 Storage 元数据）
      return t.alias();
    }"""
    cpp_src = "torch::Tensor alloc_pinned(int64_t nbytes);"

    global _alloc_pinned_module
    if _alloc_pinned_module is None:
        from torch.utils.cpp_extension import load_inline
        _alloc_pinned_module = load_inline(
            name="alloc_pinned_module",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=['alloc_pinned'],
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
        )

    nbytes = dtype.itemsize
    for dim in shape:
        nbytes *= dim
    tensor = _alloc_pinned_module.alloc_pinned(nbytes)
    tensor = tensor.view(dtype)
    return tensor.reshape(shape)


def normalize_slot_num(slot_num: int, single_slot_size: int):
    if slot_num > 0:
        return slot_num

    MAX_MEMORY_USE = 700 * 1024 * 1024 * 1024  # 700GB
    GPU_PER_MACHINE = 8
    num_slots_calculated = math.floor(MAX_MEMORY_USE / GPU_PER_MACHINE / single_slot_size)
    convert_bytes_to_gb = lambda n_bytes: round(n_bytes / 1024 / 1024 / 1024, 2)
    logging_rank_only(
        logging.warning, 0, f"auto detect num cache slots, "
        f"total memory reserved for cache: {convert_bytes_to_gb(MAX_MEMORY_USE)}, "
        f"memory for one cache slot: {convert_bytes_to_gb(single_slot_size)}, "
        f"number of cache slots: {num_slots_calculated}")
    return num_slots_calculated


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
        size_for_single_slot = layer_num * max_cache_length * kv_head_num * 2 * head_dim * kv_cache_dtype.itemsize
        self.max_cache_slot_num = normalize_slot_num(max_cache_slot_num, size_for_single_slot)
        self.max_cache_length = max_cache_length
        self.enable_paged_attn = enable_paged_attn
        self.slot_block_size = slot_block_size

        self.free_slots = set(range(self.max_cache_slot_num))
        self.cached_input_ids = torch.empty([self.max_cache_slot_num, self.max_cache_length],
                                            dtype=torch.long,
                                            device="cuda")
        self.cached_input_id_length = [0 for _ in range(self.max_cache_slot_num)]

        self.cache_slots = allocate_pinned_memory(
            [self.max_cache_slot_num, self.layer_num, self.max_cache_length, self.kv_head_num, 2, self.head_dim],
            self.kv_cache_dtype)

        # slot_id -> req_id
        self.lru_cache = collections.OrderedDict()
        self.req_id_to_slot_id = {}
        self.req_id_to_image_shift = {}

        logging_rank_only(
            logging.warning, 0, f"enable prefix cache, cache shape: {self.cache_slots.shape}, "
            f"cache size in bytes: {self.cache_slots.numel() * self.cache_slots.element_size()}, "
            f"max cached length: {max_cache_length}")

    def get_cache_ids(self):
        return self.req_id_to_slot_id.keys()

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
            del self.req_id_to_image_shift[request_id_evict]
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

    def get_image_shift(self, request_id: str):
        return self.req_id_to_image_shift.get(request_id, 0)

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

    def save_to_cache(self, request_id: str, full_input_ids: torch.Tensor, kv_cache_table: torch.Tensor, image_shift,
                      xperf_module):
        if request_id not in self.req_id_to_slot_id:
            slot_id, is_evict = self._get_free_slot(request_id)
        else:
            slot_id = self.req_id_to_slot_id[request_id]
            is_evict = False
        self.req_id_to_image_shift[request_id] = image_shift

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


@dataclass
class CacheItem(object):
    query_id: str
    input_ids: torch.Tensor = None
    input_length: int = 0
    cache_page_table: List = field(default_factory=list)


@triton.jit
def load_or_store_paged_prefix_cache_kernel(kv_cache_ptr_array, prefix_cache_ptr, kv_cache_page_table,
                                            prefix_cache_page_table, slot_block_size: tl.constexpr, kv_slot_begin,
                                            kv_slot_end, kv_total_head_num: tl.constexpr,
                                            kv_total_head_num_padded: tl.constexpr,
                                            prefix_cache_num_slots: tl.constexpr, is_load: tl.constexpr):
    # kv_cache: max_kv_cache_slot * slot_block_size * head_num * 2 * head_dim
    # prefix_cache: max_prefix_cache_slot * slot_block_size * layer_num * head_num * 2 * head_dim
    #
    # grid: [layer_num]

    layer_id = tl.program_id(0).to(tl.int64)
    block_id = tl.program_id(1).to(tl.int64)
    block_num = tl.num_programs(1).to(tl.int64)

    dtype = prefix_cache_ptr.dtype.element_ty
    kv_cache_layer_ptr = tl.load(kv_cache_ptr_array + layer_id).to(tl.pointer_type(dtype))
    prefix_cache_layer_ptr = prefix_cache_ptr + layer_id * prefix_cache_num_slots * slot_block_size * kv_total_head_num

    length = kv_slot_end - kv_slot_begin

    for pos_id in range(block_id, length, block_num):
        pos = pos_id + kv_slot_begin
        block = pos // slot_block_size
        slot_id = pos % slot_block_size

        prefix_cache_slot_idx = tl.load(prefix_cache_page_table + block).to(tl.int64)
        kv_cache_slot_idx = tl.load(kv_cache_page_table + block).to(tl.int64)

        off_kv_load = tl.arange(0, kv_total_head_num_padded).to(tl.int64)
        mask_kv = off_kv_load < kv_total_head_num

        kv_cache_base_ptr = kv_cache_layer_ptr + \
                            (kv_cache_slot_idx * slot_block_size + slot_id) * kv_total_head_num
        prefix_cache_base_ptr = prefix_cache_layer_ptr + \
                                (prefix_cache_slot_idx * slot_block_size + slot_id) * kv_total_head_num
        if is_load:
            loaded = tl.load(prefix_cache_base_ptr + off_kv_load, mask=mask_kv)
            tl.store(kv_cache_base_ptr + off_kv_load, loaded, mask=mask_kv)
        else:
            loaded = tl.load(kv_cache_base_ptr + off_kv_load, mask=mask_kv)
            tl.store(prefix_cache_base_ptr + off_kv_load, loaded, mask=mask_kv)


def load_or_store_paged_prefix_cache(kv_cache_per_layer: List[torch.Tensor], prefix_cache: torch.Tensor,
                                     kv_cache_page_table: torch.Tensor, prefix_cache_page_table: torch.Tensor,
                                     kv_slot_begin: int, kv_slot_end: int, is_load: bool):
    layer_num, max_prefix_cache_slot, slot_block_size, head_num, _, head_dim = prefix_cache.shape
    assert len(kv_cache_per_layer) == layer_num
    _, slot_block_size_, head_num_, _, head_dim_ = kv_cache_per_layer[0].shape
    assert slot_block_size == slot_block_size_
    assert head_num == head_num_
    assert head_dim == head_dim_
    kv_cache_ptr_array = torch.tensor([kv_cache.data_ptr() for kv_cache in kv_cache_per_layer],
                                      dtype=torch.long,
                                      device="cuda")

    grid = (layer_num, 64)
    assert triton.next_power_of_2(slot_block_size) == slot_block_size
    kv_total_head_num = head_num * 2 * head_dim
    kv_total_head_num_padded = triton.next_power_of_2(kv_total_head_num)

    load_or_store_paged_prefix_cache_kernel[grid](kv_cache_ptr_array, prefix_cache, kv_cache_page_table,
                                                  prefix_cache_page_table, slot_block_size, kv_slot_begin, kv_slot_end,
                                                  kv_total_head_num, kv_total_head_num_padded, max_prefix_cache_slot,
                                                  is_load)


class PagedPrefixCache(object):

    def __init__(self, kv_head_num, head_dim, layer_num, kv_cache_dtype, max_cache_slot_num, max_cache_length,
                 enable_paged_attn, slot_block_size):
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.kv_cache_dtype = kv_cache_dtype
        size_for_single_slot = layer_num * slot_block_size * kv_head_num * 2 * head_dim * kv_cache_dtype.itemsize
        self.max_cache_slot_num = normalize_slot_num(max_cache_slot_num, size_for_single_slot)
        self.max_cache_length = max_cache_length
        self.enable_paged_attn = enable_paged_attn
        self.slot_block_size = slot_block_size
        self.req_id_to_image_shift = {}

        self.free_slots = set(range(max_cache_slot_num))

        self.cache = allocate_pinned_memory([layer_num, max_cache_slot_num, slot_block_size, kv_head_num, 2, head_dim],
                                            kv_cache_dtype)

        # request_id -> CacheItem
        self.lru_cache = collections.OrderedDict()

        logging_rank_only(
            logging.warning, 0, f"enable paged prefix cache, cache shape: {self.cache.shape}, "
            f"cache size in bytes: {self.cache.numel() * self.cache.element_size()}, "
            f"max cached length: {max_cache_length}")

    def get_cache_ids(self):
        return self.lru_cache.keys()

    def clear_cache(self):
        self.free_slots = set(range(self.max_cache_slot_num))
        self.lru_cache.clear()

    def _need_to_evict(self, pages_needed: int):
        return len(self.free_slots) < pages_needed

    def _get_pages_needed(self, request_id: str, total_length: int):
        total_required = (total_length + self.slot_block_size - 1) // self.slot_block_size
        if request_id not in self.lru_cache:
            already_allocated = 0
        else:
            cache_item: CacheItem = self.lru_cache[request_id]
            already_allocated = len(cache_item.cache_page_table)
        return total_required - already_allocated

    def _get_free_slots(self, request_id: str, pages_needed: int):
        # cache_item already allocated
        cache_item: CacheItem = self.lru_cache[request_id]
        if not self._need_to_evict(pages_needed):
            pages_to_allocate = []
            for _ in range(pages_needed):
                pages_to_allocate.append(self.free_slots.pop())
            cache_item.cache_page_table.extend(pages_to_allocate)
            logging_rank_only(
                logging.warning, 0, f"allocate cache slot, request_id: {request_id}, "
                f"allocated cache slots: #{len(pages_to_allocate)}")
            return False
        else:
            # cache slot not enough
            # evict last existing slot
            evicted_request_id, evicted_cache_item = self.lru_cache.popitem(last=False)
            evicted_cache_item: CacheItem
            for page_id in evicted_cache_item.cache_page_table:
                self.free_slots.add(page_id)
            del self.req_id_to_image_shift[evicted_request_id]

            real_page_allocated = min(pages_needed, len(self.free_slots))
            pages_to_allocate = []
            for _ in range(real_page_allocated):
                pages_to_allocate.append(self.free_slots.pop())
            cache_item.cache_page_table.extend(pages_to_allocate)
            logging_rank_only(
                logging.warning, 0, f"evict cache slot, request_id: {request_id}, "
                f"allocated cache slots: #{len(pages_to_allocate)}, "
                f"evicted request_id: {evicted_request_id}, "
                f"evicted cache slots: #{len(evicted_cache_item.cache_page_table)}")
            return True

    def calc_prefix_length(self, request_id: str, full_input_ids: torch.Tensor):
        if request_id not in self.lru_cache:
            return 0
        cache_item: CacheItem = self.lru_cache[request_id]
        cached_input_ids = cache_item.input_ids
        cached_input_id_length = cache_item.input_length
        min_length = min(cached_input_id_length, full_input_ids.shape[0])
        non_zero_pos = torch.nonzero(cached_input_ids[:min_length] - full_input_ids[:min_length])
        if non_zero_pos.numel() == 0:
            prefix_length = min_length
        else:
            prefix_length = non_zero_pos.flatten()[0].item()
        return prefix_length

    def get_image_shift(self, request_id: str):
        return self.req_id_to_image_shift.get(request_id, 0)

    def load_from_cache(self, request_id: str, input_ids: torch.Tensor, kv_cache_table: torch.Tensor, xperf_module,
                        start_pos: Optional[int]):
        if request_id not in self.lru_cache:
            return 0
        cache_item: CacheItem = self.lru_cache.pop(request_id)
        self.lru_cache[request_id] = cache_item

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
            kv_cache_per_layer = [
                xperf_module._get_kv_cache(layer_id, is_cache_quant) for layer_id in range(self.layer_num)
            ]
            prefix_cache_page_table = torch.tensor(cache_item.cache_page_table).cuda()
            load_or_store_paged_prefix_cache(kv_cache_per_layer, self.cache, kv_cache_table, prefix_cache_page_table, 0,
                                             prefix_length, True)
        else:
            raise RuntimeError("non paged_attention is not supported")
        return prefix_length

    def save_to_cache(self, request_id: str, full_input_ids: torch.Tensor, kv_cache_table: torch.Tensor, image_shift,
                      xperf_module):
        if request_id not in self.lru_cache:
            cache_item = CacheItem(request_id, torch.empty([self.max_cache_length], dtype=torch.long, device="cuda"),
                                   full_input_ids.shape[0])
            self.lru_cache[request_id] = cache_item
            prefix_length = 0
        else:
            cache_item: CacheItem = self.lru_cache.pop(request_id)
            self.lru_cache[request_id] = cache_item
            prefix_length = self.calc_prefix_length(request_id, full_input_ids)
        self.req_id_to_image_shift[request_id] = image_shift
        total_length = full_input_ids.shape[0]
        if prefix_length == total_length:
            return False

        already_allocated_pages = len(cache_item.cache_page_table)
        real_total_length = min(total_length, self.max_cache_length)
        pages_needed = self._get_pages_needed(request_id, real_total_length)
        is_evict = self._get_free_slots(request_id, pages_needed)

        image_token_id = -100
        if self._get_pages_needed(request_id, real_total_length) > 0 and torch.any(full_input_ids == image_token_id):
            real_total_length = torch.nonzero(full_input_ids == image_token_id)[-1].item() + 1
            if real_total_length > self.max_cache_length:
                logging_rank_only(
                    logging.warning, 0, f"image token found in input_ids, request_id: {request_id}, "
                    f"real_total_length {real_total_length} exceeds max_cache_length {self.max_cache_length}")
                return is_evict
            while (pages_needed := self._get_pages_needed(request_id, real_total_length)) > 0:
                self._get_free_slots(request_id, pages_needed)

        real_allocated_pages = len(cache_item.cache_page_table)
        real_total_length = min(real_total_length, real_allocated_pages * self.slot_block_size)
        cache_item.input_ids[:real_total_length].copy_(full_input_ids[:real_total_length])
        cache_item.input_length = real_total_length
        logging_rank_only(
            logging.warning, 0, f"save prefix cache, request_id: {request_id}, prefix length: {prefix_length}, "
            f"total length: {total_length}, already allocated pages: #{already_allocated_pages}, "
            f"real allocated pages: #{real_allocated_pages}, real total length: {real_total_length}")
        is_cache_quant = self.kv_cache_dtype not in [torch.bfloat16, torch.float16]
        if self.enable_paged_attn:
            kv_cache_per_layer = [
                xperf_module._get_kv_cache(layer_id, is_cache_quant) for layer_id in range(self.layer_num)
            ]
            prefix_cache_page_table = torch.tensor(cache_item.cache_page_table).cuda()
            load_or_store_paged_prefix_cache(kv_cache_per_layer, self.cache, kv_cache_table, prefix_cache_page_table,
                                             prefix_length, real_total_length, False)
        else:
            raise RuntimeError("non paged_attention is not supported")
        return is_evict


def get_prefix_cache_impl(num_kv_heads, head_dim, num_layers, kv_cache_dtype, prefix_cache_impl, prefix_cache_slot_num,
                          prefix_cache_max_length, enable_paged_attn, slot_block_size):
    if prefix_cache_impl == "paged":
        return PagedPrefixCache(num_kv_heads, head_dim, num_layers, kv_cache_dtype, prefix_cache_slot_num,
                                prefix_cache_max_length, enable_paged_attn, slot_block_size)
    elif prefix_cache_impl == "normal":
        return PrefixCache(num_kv_heads, head_dim, num_layers, kv_cache_dtype, prefix_cache_slot_num,
                           prefix_cache_max_length, enable_paged_attn, slot_block_size)
    else:
        return None


if __name__ == '__main__':
    # unit test for paged prefix cache
    layer_num = 66
    slot_block_size = 1024
    head_num = 1
    head_dim = 128
    max_prefix_cache_slot = 4096
    max_kv_cache_slot = 1024
    prefix_cache = torch.zeros([layer_num, max_prefix_cache_slot, slot_block_size, head_num, 2, head_dim],
                               dtype=torch.bfloat16,
                               pin_memory=True)
    kv_cache = [
        torch.randn([max_kv_cache_slot, slot_block_size, head_num, 2, head_dim], dtype=torch.bfloat16, device="cuda")
        for _ in range(layer_num)
    ]
    check_results = True
    for length in [1024, 2048, 4096, 8192, 16384, 32768]:
        import random
        num_pages = length // slot_block_size + 1
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(5):
                kv_cache_page_table = torch.tensor(random.sample(range(max_kv_cache_slot), num_pages)).cuda()
                prefix_cache_page_table = torch.tensor(random.sample(range(max_prefix_cache_slot), num_pages)).cuda()
                kv_slot_begin = 0
                kv_slot_end = length
                load_or_store_paged_prefix_cache(kv_cache, prefix_cache, kv_cache_page_table, prefix_cache_page_table,
                                                 kv_slot_begin, kv_slot_end, True)
        events = {item.key: item for item in p.key_averages()}
        event = events['load_or_store_paged_prefix_cache_kernel']
        print(length, event.cuda_time)

        if check_results:
            for layer in range(layer_num):
                prefix_cache_ = prefix_cache[layer]
                kv_cache_ = kv_cache[layer]
                for pos in range(kv_slot_begin, kv_slot_end):
                    slot_id = pos // slot_block_size
                    offset = pos % slot_block_size
                    prefix_cache_sliced = prefix_cache_[prefix_cache_page_table[slot_id], offset]
                    kv_cache_sliced = kv_cache_[kv_cache_page_table[slot_id], offset]
                    torch.testing.assert_close(prefix_cache_sliced, kv_cache_sliced.cpu())
