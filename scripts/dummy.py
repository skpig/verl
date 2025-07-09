#!/usr/bin/env python
# torch_dummy_burn.py
"""
Dummy GPU burner – 直接配合 `torchrun --standalone` 使用。
示例（单机 16×L20）：
  torchrun --standalone --nproc_per_node=16 \
           torch_dummy_burn.py --matrix-size 8192 --duration 300
"""

import os, time, argparse, torch
from torch.distributed import init_process_group

def main():
    # ---------- CLI ----------
    parser = argparse.ArgumentParser("GPU burner (torchrun flavour)")
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="边长 N：做 N×N × N×N 的 matmul")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="张量数据类型")
    parser.add_argument("--duration", type=int, default=300,
                        help="运行时长（秒）")
    args = parser.parse_args()

    # ---------- torchrun 环境变量 ----------
    local_rank = int(os.environ["LOCAL_RANK"])     # 0-based, 每节点内序号
    global_rank = int(os.environ["RANK"])          # 全局序号，可用于打印
    world_size  = int(os.environ["WORLD_SIZE"])    # 全局进程数

    # ---------- 设备 & 进程组 ----------
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_process_group(backend="nccl")             # env:// + 上述变量即可

    # ---------- 申请20G显存的tensor ----------
    m = 40 * 1024 * 1024 * 1024 // 8 # 20G显存的tensor，每个元素8字节
    a = torch.randn((m,), device=device, dtype=torch.float32)

    # ---------- 常驻张量 ----------
    m = args.matrix_size
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    a = torch.randn((m, m), device=device, dtype=dtype)
    b = torch.randn_like(a)

    # ---------- 预热 ----------
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    # ---------- 主循环 ----------
    start = time.time()
    iters = 0
    # while time.time() - start < args.duration:
    while True:
        _ = torch.matmul(a, b)
        iters += 1
        if iters % 100 == 0:
            torch.cuda.synchronize(device)

    torch.cuda.synchronize(device)
    if global_rank == 0:
        elapsed = time.time() - start
        print(f"[Done]  total iters/GPU: {iters:,}  "
              f"time: {elapsed:.1f}s  "
              f"speed: {iters/elapsed:.1f} it/s")

if __name__ == "__main__":
    main()
