cmd="docker build  \
    --build-arg ALL_PROXY=${all_proxy} \
    -t tool-rl-sglang ."
echo ${cmd}
eval ${cmd}