docker run \
    -d \
    -it \
    --shm-size 64g \
    --gpus all \
    -v /home/huangbz/.tmux.conf:/root/.tmux.conf \
    -v /home/huangbz/.config/clash:/root/.config/clash \
    -v /home/huangbz/clash:/root/clash \
    -v /home/huangbz/verl/.cache:/root/verl/.cache \
    -v /home/huangbz/verl/checkpoints:/root/verl/checkpoints \
    -v /home/huangbz/.cache/docker:/root/.cache \
    -v ${MY_DATA_DIR}:/data \
    -v ${MY_MODEL_DIR}:/pretrain \
    --ipc=host \
    --net=host \
    --privileged \
    --name sglang_container \
    ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5 \
    bash