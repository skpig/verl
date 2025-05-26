# First make sure the now() function is available in current shell
# Create logs directory if it doesn't exist

# Set GPUs and run with better log organization
export CUDA_VISIBLE_DEVICES=2,3
# Define a timestamp function
function now() {
    date '+%Y-%m-%d-%H-%M'
}

bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh trainer.experiment_name=qwen2.5-3b_rm-gsm8k-sgl-multiturn-$(now)
# bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh trainer.experiment_name=qwen2.5-3b_rm-gsm8k-sgl-multiturn-$(now) 2>&1 | tee logs/gsm8k-$(now).log  &
# nohup bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh trainer.experiment_name=qwen2.5-3b_rm-gsm8k-sgl-multiturn-$(now) > logs/gsm8k-$(now).log 2>&1 &