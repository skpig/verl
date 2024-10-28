from xperf_gpt.inference.session import InferenceSession
import xperf_gpt
import torch
import logging
import torch.nn.functional as F



def offload_to_cpu(tp_model):
    param_list = [tp_model.layernorm_weight, tp_model.wte_weight, tp_model.lm_head_weight] + \
                    [p for layer in tp_model.layers_weight for p in layer if isinstance(p, torch.Tensor)]
    if hasattr(tp_model, 'wpe'):
        param_list.append(tp_model.wpe.weight)
    for param in param_list:
        param.data = torch.empty_like(param.data, device='cpu')

    if hasattr(tp_model, '_free_kv_cache'):
        tp_model._free_kv_cache()

    torch.cuda.empty_cache()

# torchrun --nproc_per_node 1 --master_port 9898 xperf_cuda_graph.py


if __name__ == "__main__":
    # xperf_gpt.load_xperf_gpt()
    path='./modules/libModules.so'
    torch.classes.load_library(path)

    max_length = 256
    max_new_tokens = 128
    max_batch_size = 128
    num_slots = max_batch_size  # orca
    inference_sess = InferenceSession(
        num_slots=num_slots,
        max_batch_size=max_batch_size,
        max_length=max_length,
        slot_block_size=256,
        use_vllm=False,
        context_split_len=1024,
        enable_cuda_graph=False, # enable cuda graph
    )
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=1,
        top_p=0.7,
        temperature=1.0,
        rank0_split_backend="nccl"
    )
    inference_sess.init_inference_engine(
        "/opt/tiger/alpha-seed/tests/infer/config.json", generate_kwargs, enable_metrics=True
    )
    query_pool = ["猪能吃钛合金吗？", "你好"]
    timeout_seconds = 2
    num_bon = 5
    import time
    torch.cuda.synchronize()
    start = time.time()
    inference_sess.execute(query_pool, None, timeout_seconds, num_bon)
    for v in inference_sess.get_inorder_responses():
        if v.output_prompt is not "":
            print(f"idx: {v.idx} input: {v.input_prompt}\noutput: {v.output_prompt}")
        else:
            print(f"idx: {v.idx} input: {v.input_prompt}\nunfinished timeout output: {v.new_token_ids}")
    torch.cuda.synchronize()
    print(f"Time spent: {time.time()-start}")

    print("torch.cuda.memory_allocated: {} MB".format(torch.cuda.memory_allocated()/1024/1024))

    offload_to_cpu(inference_sess.engine.module)
    print("offloaded, torch.cuda.memory_allocated: {} MB".format(torch.cuda.memory_allocated()/1024/1024))