echo 'source ~/.python/verl-multiturn-rollout/bin/activate' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export WANDB_API_KEY="11e12737d5a4883f53fd55089f321f01171ffc5a"' >> ~/.bashrc
echo 'export MY_MODEL_DIR=/pretrain/' >> ~/.bashrc
echo 'export MY_DATA_DIR=/data/' >> ~/.bashrc

# apt update
# apt install -y python3.10 python3.10-venv
# python3 -m ensurepip --upgrade
# # Create a virtual environment
# python3 -m venv ~/.python/verl-multiturn-rollout

# # Activate the virtual environment
# source ~/.python/verl-multiturn-rollout/bin/activate

# cd ~
# git clone https://github.com/skpig/verl.git
# cd verl
# git fetch upstream
# git pull upstream main

# Install networx
apt update
sudo apt-get install graphviz graphviz-dev
pip install networkx[default,extra]

# Install uv
python3 -m pip install uv

# Install SGLang
python3 -m uv pip install -e ".[sglang]"

# Manually install flash-attn
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps

# Install verl
python3 -m uv pip install .
python3 -m uv pip install -r ./requirements.txt


# Define a timestamp function
function now() {
    date '+%Y-%m-%d-%H-%M'
}