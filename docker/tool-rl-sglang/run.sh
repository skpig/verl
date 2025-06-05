docker run \
    -d \
    -it \
    --shm-size 32g \
    --gpus all \
    -v /home/huangbz/.config/clash:/root/.config/clash \
    -v /home/huangbz/clash:/root/clash \
    -v /home/huangbz/verl/.cache:/root/verl/.cache \
    -v /home/huangbz/verl/checkpoints:/root/verl/checkpoints \
    -v /home/huangbz/.cache/docker:/root/.cache \
    -v ${MY_DATA_DIR}:/data \
    -v ${MY_MODEL_DIR}:/pretrain \
    --ipc=host \
    --network=host \
    --add-host=host.docker.internal:host-gateway \
    --privileged \
    --name vineppo \
    tool-rl-sglang \
    /bin/zsh