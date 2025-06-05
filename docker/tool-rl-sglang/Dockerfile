# 使用 lmsysorg/sglang:dev 作为基础镜像
FROM lmsysorg/sglang:dev

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/root/.python/verl-multiturn-rollout
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 安装必要的系统依赖和 Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        build-essential \
        python3.10 \
        python3.10-venv \
        python3.10-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# 创建并激活虚拟环境
RUN python3.10 -m venv $VIRTUAL_ENV

# 设置工作目录
WORKDIR /root

# 克隆 verl 仓库
RUN git clone https://github.com/skpig/verl.git && \
    cd verl && \
    git remote add upstream https://github.com/skpig/verl.git && \
    git fetch upstream && \
    git pull upstream main

ADD init.sh /root/verl/init.sh

# 设置默认命令
CMD ["/bin/zsh"]
