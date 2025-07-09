from pathlib import Path
import pytest
import copy
import time

from omegaconf import OmegaConf

from mono_rl import DataProto
import pandas as pd
import numpy as np
import uuid
from tests.test_utils import gpu_allocator, ray_fixture, set_common_envs, get_config, get_tokenizer, create_rollout_manager, PytestXdistEnv


def get_dataproto(config, tokenizer):
    question_prompt = (
        "We define a new math operator @, where you can only call an external tool to compute. Please "
        "put your final answer inside \\boxed{} only in the last turn. Now answer the following question")
    qa_list = [
        (f"{question_prompt}: Compute (3 @ 3) @ 1 @ 7", "7"),
        (f"{question_prompt}:\nCompute 4 @ 6 @ 4", "-8"),
        (f"{question_prompt}:\nCompute 5 @ 3", "9"),
        (f"{question_prompt}:\nCompute 8 @ 1", "22"),
    ]

    data = []
    for question, answer in qa_list:
        prompt = [{"role": "user", "content": question}]
        reward_model = {'style': 'rule-lighteval/MATH', 'ground_truth': str(answer)}
        data.append({"prompt": prompt, "reward_model": reward_model})
    df = pd.DataFrame(data)

    sentences = tokenizer.apply_chat_template(
        [prompt for prompt in df.prompt.tolist()],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_data = tokenizer(
        sentences,
        padding="max_length",
        return_tensors="pt",
        padding_side='left',
        max_length=config.data.max_prompt_length,
    )

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    batch = DataProto.from_dict(data)
    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch.non_tensor_batch['reward_model'] = np.array(df['reward_model'].tolist(), dtype=object)
    batch.non_tensor_batch['raw_prompt'] = np.array([[{"role": "user", "content": q}] for q in qa_list], dtype=object)
    batch.check_consistency()
    return batch


def get_common_config():
    override_config = OmegaConf.create({
        "data": {
            "train_batch_size": 4,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
            },
            "rollout": {
                "tensor_model_parallel_size": 2,
                "mode": "batch",
                "weights_communicator": "nccl",
                "complete_ratio": 1.0,
                "rollout_pool": {
                    "warmup_step": 0,
                },
                "gpu_memory_utilization": 0.4
            },
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 2,
            "project_name": "alpha_seed_test",
            "experiment_name": "rollout_manager",
            "logger": ['console'],
        },
        "streaming_rollout": {
            "nnodes": 0,
            "n_gpus_per_node": 2,
        },
        "streaming_validator": {
            "nnodes": 0,
            "n_gpus_per_node": 2,
        },
    })
    return get_config(override_config)


def decode_output(output, tokenizer):
    output_ids = output.batch['input_ids']
    text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return text_out


def _check_score(out_text, batch):

    def reward_fn(solution_str: str, ground_truth: str):
        return 1.0 if ground_truth in solution_str else 0.0

    total_score = 0
    for text, item in zip(out_text, batch.chunk(len(batch))):
        reward_model = item.non_tensor_batch['reward_model'][0]
        total_score += reward_fn(solution_str=text, ground_truth=reward_model['ground_truth'])
    assert total_score >= len(out_text) - 1, \
        f"allow only 1 wrong answer, full score: {len(out_text)}, got: {total_score}"


def mock_save_dataproto(data: DataProto, prefix: str = ''):
    print(f"mock savedataproto prefix={prefix}")


# python3 -m pytest /opt/tiger/alpha-seed/tests/rollout/test_rollout_manager_agent.py::test_train_generate\[4-False-ucx-True-1.0\] -x --pdb -s --pdbcls=IPython.terminal.debugger:TerminalPdb


@pytest.mark.parametrize("complete_ratio", [1.0, 0.0, 0.5])
@pytest.mark.parametrize("is_server", [True])
@pytest.mark.parametrize("weights_communicator", ["nccl", "ucx"])
@pytest.mark.parametrize("elastic", [True, False])
@pytest.mark.parametrize("gpu_allocator", [4], indirect=True)
def test_train_generate(set_common_envs, gpu_allocator, ray_fixture, complete_ratio, is_server, weights_communicator,
                        elastic):
    if elastic:
        pytest.skip("temporarily disabled due to underlying ray state API conflict with multiple clusters")
    if elastic and weights_communicator == "nccl":
        pytest.skip("skip elastic and weights_communicator=nccl")
    if elastic and complete_ratio > 0.0:
        pytest.skip("skip elastic and complete_ratio > 0.0")

    config = get_common_config()
    has_standalone = complete_ratio < 1.0
    config.actor_rollout_ref.rollout.mode = 'server' if is_server else 'batch'
    config.actor_rollout_ref.rollout.weights_communicator = weights_communicator
    config.actor_rollout_ref.rollout.complete_ratio = complete_ratio
    config.streaming_rollout.elastic.enable = elastic
    config.streaming_rollout.elastic.min_replicas = 1
    config.streaming_rollout.elastic.max_replicas = 1  # 资源有限，这个UT里先不扩
    config.streaming_rollout.nnodes = 1 if has_standalone else 0
    config.streaming_validator.nnodes = 0

    # agent related config
    config.actor_rollout_ref.model.path = 'hdfs://harunava/home/byte_data_seed_azure/alphaseed/daiweinan/models/qwen3_0.6b_p6d'
    config.data.max_prompt_length = 8192
    config.data.max_response_length = 8192
    config.data.return_raw_chat = True
    config.actor_rollout_ref.rollout.agent.max_turns = 10
    config.actor_rollout_ref.rollout.agent.max_new_tokens_per_turn = 2048
    config.data.chat_template = 'chatml_tool'
    config.data.dataloader_raw_template = True
    config.reward_model.last_characters = 300
    config.rollout_server.handler = 'agent/tool/special_calculator'
    config.algorithm.use_model_output_mask = True

    tokenizer = get_tokenizer(config)
    if config.data.get('chat_template', None) == 'raw':
        raw_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
        tokenizer.chat_template = raw_template
        if tokenizer.bos_token is None:
            tokenizer.bos_token = ""
    if config.data.get('chat_template', None) == 'chatml':
        # chatml from https://huggingface.co/docs/transformers/v4.53.1/en/chat_templating
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    if config.data.get('chat_template', None) == 'chatml_tool':
        tokenizer.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if tools %}{{ '<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}{%- for tool in tools %}{{- '\n' }}{{ tool | tojson }}{%- endfor %}\n\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n{% endif %}{% for message in messages %}{% if message['role'] == 'tool' %}<|im_start|>user\n<tool_response>\n{{ message['content'] }}\n</tool_response><|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}\n{% else %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""

    batch = get_dataproto(config, tokenizer)
    rollout_manager = create_rollout_manager(config)

    input_batch = copy.deepcopy(batch)
    warmup_elapsed = 0
    try:
        for i in range(3):
            start = time.time()
            batch = copy.deepcopy(input_batch)
            is_warmup_step = i == 0
            batch = rollout_manager.train_generate(batch,
                                                   step=i,
                                                   save_dataproto_fn=mock_save_dataproto,
                                                   is_warmup_step=is_warmup_step)
            if is_warmup_step:
                warmup_elapsed = time.time() - start
                assert batch is None
            else:
                out_text = decode_output(batch, tokenizer)
                print(out_text)
                _check_score(out_text, batch)
                if complete_ratio < 1.0:
                    # mock training to overlap
                    time.sleep(warmup_elapsed)
    except:
        raise
    finally:
        rollout_manager.stop_servers()


@pytest.mark.parametrize("is_standalone", [True, False])
@pytest.mark.parametrize("is_server", [True, False])
@pytest.mark.parametrize("weights_communicator", ["nccl", "ucx"])
@pytest.mark.parametrize("gpu_allocator", [4], indirect=True)
def test_val_generate(set_common_envs, gpu_allocator, ray_fixture, is_standalone, is_server, weights_communicator):
    if not is_standalone and weights_communicator == "ucx":
        pytest.skip("skip weights_communicator=ucx and hybrid engine mode")

    config = get_common_config()
    config.actor_rollout_ref.rollout.mode = 'server' if is_server else 'batch'
    config.actor_rollout_ref.rollout.weights_communicator = weights_communicator
    config.streaming_rollout.nnodes = 0
    config.streaming_validator.nnodes = 1 if is_standalone else 0

    tokenizer = get_tokenizer(config)
    batch = get_dataproto(config, tokenizer)
    rollout_manager = create_rollout_manager(config)
    try:
        batch = rollout_manager.val_generate(batch, is_async=is_standalone)
        out_text = decode_output(batch, tokenizer)
        print(out_text)
        _check_score(out_text, batch)
    except:
        raise
    finally:
        rollout_manager.stop_servers()


@pytest.mark.parametrize("is_server", [True, False])
@pytest.mark.parametrize("train_standalone", [True, False])
@pytest.mark.parametrize("val_standalone", [True, False])
@pytest.mark.parametrize("gpu_allocator", [4], indirect=True)
def test_streaming_train_val(set_common_envs, gpu_allocator, ray_fixture, is_server, train_standalone, val_standalone):
    config = get_common_config()
    config.actor_rollout_ref.rollout.mode = 'server' if is_server else 'batch'
    config.actor_rollout_ref.rollout.complete_ratio = 0.0 if train_standalone else 1.0
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.streaming_rollout.nnodes = 1 if train_standalone else 0
    config.streaming_rollout.n_gpus_per_node = 1
    config.streaming_validator.nnodes = 1 if val_standalone else 0
    config.streaming_validator.n_gpus_per_node = 1

    tokenizer = get_tokenizer(config)
    rollout_manager = create_rollout_manager(config)
    import threading
    val_stop = threading.Event()
    val_batch = get_dataproto(config, tokenizer)

    # run warmup step
    start = time.time()
    input_batch = get_dataproto(config, tokenizer)
    _ = rollout_manager.train_generate(copy.deepcopy(input_batch),
                                       step=0,
                                       save_dataproto_fn=mock_save_dataproto,
                                       is_warmup_step=True)
    warmup_time = time.time() - start

    def val_thread_fn():
        val_step = 0
        while not val_stop.is_set():
            input_batch = copy.deepcopy(val_batch)
            batch = rollout_manager.val_generate(input_batch, is_async=val_standalone)
            out_text = decode_output(batch, tokenizer)
            print("val:", out_text)
            _check_score(out_text, batch)
            val_step += 1

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)

    val_fut = executor.submit(val_thread_fn)

    try:
        for i in range(1, 3):
            start = time.time()
            batch = copy.deepcopy(input_batch)
            batch = rollout_manager.train_generate(batch,
                                                   step=i,
                                                   save_dataproto_fn=mock_save_dataproto,
                                                   is_warmup_step=False)
            out_text = decode_output(batch, tokenizer)
            print("train:", out_text)
            _check_score(out_text, batch)
            # mock training to overlap
            time.sleep(warmup_time + 1)
    except:
        raise
    finally:
        val_stop.set()
        val_fut.result()
        rollout_manager.stop_servers()
