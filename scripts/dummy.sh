# 单节点 16 GPU
torchrun --standalone --nproc_per_node=16 \
         scripts/dummy.py --matrix-size 8192 --duration 300
