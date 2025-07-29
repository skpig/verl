# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import yaml
import warnings
import contextlib
from typing import Union, List
from collections import defaultdict
import tempfile
import random
import numpy as np
from codetiming import Timer

from mono_rl import DataProto
import torch
from verl.utils.tracking import Tracking
import wandb
import os
import pandas as pd
import hdfs_io
from datetime import datetime
from multiprocessing import Process

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from alpha_seed.trainer.ppo import RayPPOTrainer
from alpha_seed.utils.duplicate import para_dup
from alpha_seed.utils.dataset.rl_dataset import collate_fn, RLHFDataset
from alpha_seed.workers.actors.checkpoint import CkptGlobalUploader
from alpha_seed.workers.streaming_service.streaming_utils import record_xperf_metrics
from alpha_seed.utils.server_client import ClientTaskRunner, ServerHealthCheck, recreate_actor
from tasks.main_ppo import validate_config, RewardManager
from omegaconf import OmegaConf, ListConfig, DictConfig
import ray
import hydra
from verl.utils.fs import copy_local_path_from_hdfs
from hdfs_io.hdfs_io import hcopy, hmkdir

import re
import signal
from typing import Optional

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem":
                "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution":
                "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot":
                "1",
        },
        {
            "problem":
                "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution":
                "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot":
                "1",
        },
        {
            "problem":
                "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution":
                "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot":
                "1",
        },
        {
            "problem":
                "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution":
                "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot":
                "1",
        },
    ]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


class timeout:

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=10):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                    sympy.parsing.latex.errors.LaTeXParsingError,
                    sympy.SympifyError,
                    TypeError,
            ):
                # eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                # eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                # eval_logger.debug(
                # f"Had some trouble simplifying when comparing {x1} and {x2}"
                # )
                return False

    except TimeoutError:
        # eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        # eval_logger.error(e)
        raise
    except Exception as e:
        # eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    # "ft", #this is dangerous, infty, left will be damaged!
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


INVALID_ANS_GSM8k = "[invalid]"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


def filter_ignores(st, regexes_to_ignore):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st


def is_correct_integer(
    og_pred,
    gt,
):
    numbers = re.findall(r'-?\d+', og_pred)
    numbers = numbers[-1] if len(numbers) > 0 else ""  # 很难通过枚举把最后一个搞成正确答案
    correctness = gt == numbers
    return correctness, og_pred


def is_correct_minerva(og_pred, gt, gt_need_extract=False):
    match = re.findall(ANSWER_PATTERN, og_pred)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)
    # return (pred == gt or is_equiv(pred, gt)), pred
    return (pred == gt), pred


def compute_score(
    pred,
    answer,
):
    """
    default行为：对给1，其余给-1
    punish_no_answer:
    * v0: 0
    * v1: -0.1
    * v2: -0.2
    """
    # breakpoint()
    corr_minerva, pred_minerva = is_correct_minerva(pred,
                                                    answer)  # To remove if math is also converted to interger format
    corr_integer, pred_integer = is_correct_integer(pred, answer)
    pred = pred_minerva if corr_minerva else pred_integer
    corr = corr_minerva or corr_integer

    reward = 1 if corr else 0
    return reward


class SimpleDataset(Dataset):

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 cache_dir="~/.cache/alphaseed/gen_cli",
                 truncation='error',
                 preprocess_mode='RAW'):

        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.preprocess_mode = preprocess_mode

        self._download()
        self._read_files()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)
            print(i, self.parquet_files[i])

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        import verl.utils.torch_functional as verl_F
        row_dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict[self.prompt_key]
        # Apply chat template here, align with seed/cook
        if self.preprocess_mode == 'RAW':
            pass
        elif self.preprocess_mode == 'CHATML_SESSION':
            chat = f"{self.tokenizer.bos_token}user\n{chat}{self.tokenizer.eos_token}{self.tokenizer.bos_token}assistant\n"
        else:
            raise ValueError(f"unsupported preprocess_mode: {self.preprocess_mode}")

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=chat,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        row_dict['input_ids'] = input_ids[0].to(torch.int32)
        row_dict['attention_mask'] = attention_mask[0].to(torch.int8)
        row_dict['off_policy_steps'] = torch.zeros([1]).to(torch.int8)
        return row_dict


def compute_score_by_rule(data):
    output_score = {}
    for k, conts in data.items():
        scores = []
        for i in range(len(conts)):
            if isinstance(conts[i]['output'], str):
                outputs = [conts[i]['output']]
            else:
                outputs = conts[i]['output']
            for pred in outputs:
                answer = conts[i]['answer']
                score = compute_score(pred, str(answer))
                scores.append(score)
        output_score[k] = scores
    return output_score


def sample_and_compute_score(df: pd.DataFrame, bon_list: List[int], sample_num: int):
    df = df.sample(frac=1)
    print(len(df))
    data = defaultdict(list)

    for _, line in df.iterrows():
        id = line['id']
        data[id].append(line)
    print(len(data))

    rule_scores = compute_score_by_rule(data)

    random.seed(2024)
    bok_list = []
    for k in bon_list:
        assert sample_num >= k, f"{sample_num} < {k}"
        bok = 0
        for i in range(100):
            for idx, scores in rule_scores.items():
                select_score = random.sample(scores[:sample_num], k)
                if sum(select_score) >= 1:
                    bok += 1
        bok_list.append(bok / 100 / 30)

    print(bok_list)


def slice_data_proto(batch: DataProto, slice_num: int):
    sliced = batch[:slice_num]
    return DataProto(batch=sliced.batch, non_tensor_batch=sliced.non_tensor_batch, meta_info=sliced.meta_info)


class GenClient:

    def __init__(self, config, kv_store_name="kv_store"):
        self.config = config
        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(self.config.data.tokenizer)
        # instantiate tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.kv_store = ray.get_actor(kv_store_name)
        server_config = ray.get(self.kv_store.get_by_key.remote('config'))
        client_config = server_config
        client_config.server_client.role = "client"

        server_tokenizer = ray.get(self.kv_store.get_by_key.remote('tokenizer'))

        resource_pool_manager = ray.get(self.kv_store.get_by_key.remote('resource_pool_manager'))
        ray_worker_group_cls = ray.get(self.kv_store.get_by_key.remote('ray_worker_group_cls'))

        role_worker_mapping = ray.get(self.kv_store.get_by_key.remote('role_worker_mapping'))
        available_roles = list(role_worker_mapping.keys())

        print(f"The server provides roles: {', '.join([str(role) for role in available_roles])}")

        self.dataloader = self._create_dataloader()

        self.logger = Tracking(project_name=server_config.trainer.project_name,
                               experiment_name=server_config.trainer.experiment_name,
                               default_backend=server_config.trainer.logger,
                               config=OmegaConf.to_container(config, resolve=True))

        self.trainer = RayPPOTrainer(config=client_config,
                                     tokenizer=server_tokenizer,
                                     role_worker_mapping=role_worker_mapping,
                                     resource_pool_manager=resource_pool_manager,
                                     ray_worker_group_cls=ray_worker_group_cls,
                                     reward_fn=None,
                                     val_reward_fn=None,
                                     logger=self.logger)

    def _create_dataloader(self):
        from torch.utils.data import SequentialSampler
        if self.config.data.format == "simple":
            assert self.config.data.answer_key == "answer", "answer_key=answer required for this dataset format"
            dataset = SimpleDataset(parquet_files=self.config.data.input_files,
                                    tokenizer=self.tokenizer,
                                    prompt_key=self.config.data.prompt_key,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    truncation=self.config.data.truncation,
                                    preprocess_mode="CHATML_SESSION")

        elif self.config.data.format == "rl":
            from mono_rl.utils.seed import CHAT_TEMPLATE
            self.tokenizer.chat_template = CHAT_TEMPLATE
            dataset = RLHFDataset(parquet_files=self.config.data.input_files,
                                  tokenizer=self.tokenizer,
                                  prompt_key=self.config.data.prompt_key,
                                  answer_key=self.config.data.answer_key,
                                  use_ref_answer=False,
                                  max_prompt_length=self.config.data.max_prompt_length,
                                  max_response_length=self.config.data.max_response_length,
                                  filter_prompts=True,
                                  return_raw_chat=self.config.data.get('return_raw_chat', False),
                                  truncation=self.config.data.get('truncation', 'error'),
                                  multi_prompts=self.config.data.get("multi_prompts", "none"),
                                  num_prompts_per_data=self.config.data.get("num_prompts_per_data", 1))
        else:
            raise ValueError(f"unsupported data format {self.config.data.format}")

        if self.config.data.total_num > 0:
            from torch.utils.data import Subset
            dataset = torch.utils.data.Subset(dataset, indices=range(0, self.config.data.total_num))
        sampler = SequentialSampler(data_source=dataset)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.config.gen.batch_size,
                                shuffle=None,
                                drop_last=False,
                                collate_fn=collate_fn,
                                sampler=sampler)
        return dataloader

    def init_workers(self):
        self.trainer.init_workers(kv_store=self.kv_store, ckpt_global_uploader=None)
        self.actor_rollout_wg = self.trainer.actor_rollout_wg

    def gen(self):
        total_iters = len(self.dataloader)
        data = []
        gen_bs = self.config.gen.batch_size
        for iter, batch_dict in enumerate(self.dataloader):
            print(f"Running iter #{iter}/{total_iters}...")
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            origin_bs = len(batch)

            if origin_bs < gen_bs:
                # padding to batch_size
                repeat_num = (gen_bs + origin_bs - 1) // origin_bs
                batch = batch.repeat(repeat_num, interleave=False)
                batch = slice_data_proto(batch, gen_bs)

            if 'id' not in batch.non_tensor_batch:
                batch.non_tensor_batch['id'] = np.array(list(range(iter * gen_bs, (iter + 1) * gen_bs)), dtype=object)

            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'off_policy_steps'])
            gen_batch.meta_info.update({'generation_kwargs': self.config.gen.generate_kwargs, 'complete_ratio': 1.0})
            metrics = dict()
            with Timer(name='gen', logger=None) as timer:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            metrics['timing/gen'] = timer.last
            record_xperf_metrics(gen_batch_output, metrics, self.logger, iter, prefix='hybrix')

            if len(gen_batch_output) > origin_bs:
                # remove padding
                batch = slice_data_proto(batch, origin_bs)
                gen_batch_output = slice_data_proto(gen_batch_output, origin_bs)

            input_ids = gen_batch_output.batch['input_ids']
            prompt_ids = input_ids[:, :self.config.data.max_prompt_length]
            response_ids = input_ids[:, self.config.data.max_prompt_length:]

            first_non_one_indices = (prompt_ids != self.tokenizer.pad_token_id).int().argmax(dim=1)
            rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]

            for i in range(len(batch)):
                item = {
                    'id': batch.non_tensor_batch['id'][i],
                    self.config.data.prompt_key: self.tokenizer.decode(rmv_padding_prompt_ids[i]),
                    self.config.data.answer_key: batch.non_tensor_batch['answer'][i],
                    'output': self.tokenizer.decode(response_ids[i, :], skip_special_tokens=True),
                }
                data.append(item)

            self.logger.log(data=metrics, step=iter)

        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(mode='w', suffix=".parquet") as f:
            df.to_parquet(f.name)
            hmkdir(os.path.dirname(self.config.data.output_file))
            hcopy(f.name, self.config.data.output_file)
        return df

    def gen_and_eval(self):
        df = self.gen()
        sample_and_compute_score(df, self.config.eval.bon_list, self.config.eval.sample_num)


def main_task(config: DictConfig):
    gen_cli = GenClient(config)
    gen_cli.init_workers()
    gen_cli.gen_and_eval()


@hydra.main(config_path='config', config_name='gen_client', version_base=None)
def main(config):
    if config.gen.skip:
        local_path = copy_local_path_from_hdfs(config.data.output_file)
        df = pd.read_parquet(local_path)
        sample_and_compute_score(df, config.eval.bon_list, config.eval.sample_num)
        return

    alpha_seed_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.join(alpha_seed_root, "tasks/runtime_env/runtime_env.yaml")) as fin:
        runtime_env = yaml.safe_load(fin)

    ray_is_ready = False
    for _ in range(600):
        try:
            ray.init(namespace="alphaseed", address=config.ray.server_addr, runtime_env=runtime_env)
            ray_is_ready = True
            break
        except:
            print("waiting for ray server init...")
            time.sleep(1)
    if not ray_is_ready:
        raise RuntimeError("wait for ray cluster ready timeout")

    is_server_ready = False
    for i in range(200):
        try:
            server_health_check = ray.get_actor(ServerHealthCheck.name)
            is_server_ready = ray.get(server_health_check.is_ready.remote())
            if is_server_ready:
                break
        except Exception as e:
            print(f"waiting for server to be ready (iter #{i})...: [{e}]")
            time.sleep(5)

    if not is_server_ready:
        raise RuntimeError("wait for server ready timeout")

    runner = recreate_actor(ClientTaskRunner, name=ClientTaskRunner.name)
    ray.get(runner.main.remote(main_task, config=config))


if __name__ == '__main__':
    main()
