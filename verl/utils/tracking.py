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
"""
A unified tracking interface that supports logging data to different backend
"""

from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
import dataclasses
import multiprocessing
import os
from enum import Enum
from functools import partial
from pathlib import Path
import time
from typing import Any, Dict, List, Union

import wandb

from verl import DataProto

def make_bytes_char():
    bs = []
    # Add characters from '!' to '~' (ASCII 33 to 126)
    bs.extend(range(ord('!'), ord('~') + 1))
    # Add characters from '\xA1' to '\xAC' (ASCII 161 to 172)
    bs.extend(range(0xA1, 0xAD))
    # Add characters from '\xAE' to '\xFF' (ASCII 174 to 255)
    bs.extend(range(0xAE, 0x100))
    # Create a list of Unicode values (UTF-32)
    cs = [b for b in bs]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # Create a dictionary mapping Unicode characters to bytes
    char_to_byte = {}
    for i in range(len(bs)):
        l = chr(cs[i])
        r = chr(bs[i])
        if l != r:
            char_to_byte[l] = r

    byte_translate_map = str.maketrans(char_to_byte)
    return byte_translate_map

BYTE_TRANSLATE_MAP = make_bytes_char()

def decode_worker_init(tokenizer_name_or_path, padding_side):
    # to avoid serializing/deserializing the tokenizer object from the main process
    # child process initializes it's own tokenizer
    global child_tokenizer
    child_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side=padding_side)


def clean_up_special_token(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # fixing the issue here: https://github.com/QwenLM/Qwen2.5/issues/834
    tokens = [t.translate(BYTE_TRANSLATE_MAP) if t else t for t in tokens]
    return tokens


def decode_response(prompt, response):
    prompt = prompt[prompt > 0]
    decoded_prompt = child_tokenizer.decode(prompt, skip_special_tokens=True)
    decoded_response = child_tokenizer.decode(response, skip_special_tokens=True)
    decoded_response_clean = clean_up_special_token(child_tokenizer, response)
    return decoded_prompt, decoded_response, decoded_response_clean

def async_tracking_log_samples(train_batch, tokenizer, global_step):
    responses = train_batch.batch["responses"]
    batch_size, response_length = responses.shape
    print(time.ctime(), "sample shape", responses.shape)

    select_keys = [
        "old_log_probs", "entropys", "returns", "values", "advantages", "token_level_rewards"
    ]
    real_response_lens = train_batch.batch['attention_mask'][:, -response_length:].numpy().sum(-1).tolist()
    # raw_scores = train_batch.batch["raw_scores"].numpy().sum(-1).tolist()
    print(time.ctime(), "sample tolist done")
    samples = [None for i in range(batch_size)]

    max_workers = max(32, multiprocessing.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=decode_worker_init,
                             initargs=(tokenizer.name_or_path, tokenizer.padding_side)) as executor:
        for i in range(batch_size):
            future = executor.submit(decode_response, train_batch.batch["prompts"][i], responses[i])

            per_token_info = {}
            for k in select_keys:
                if k in train_batch.batch:
                    v = train_batch.batch[k][i].tolist()
                    assert len(v) == response_length, f"Metrics[{k}] must match response_length"
                    per_token_info[k] = v

            sample_info = {
                "raw_score": int(train_batch.non_tensor_batch['score'][i]),
                "response_length": real_response_lens[i],
            }
            samples[i] = [future, per_token_info, sample_info]

    rl_samples = [None for i in range(batch_size)]
    for i, item in enumerate(samples):
        decoded_prompt, decoded_response, decoded_response_clean = item[0].result()
        sample = wandb.RlSample(decoded_prompt, decoded_response, decoded_response_clean, *item[1:])
        rl_samples[i] = sample

    print(time.ctime(), "sample to RlSample done")
    wandb.log({"train_samples": rl_samples}, step=global_step)
    print(time.ctime(), "sample wandb.log done")


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "bwandb", "clearml"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None, resume_step=0):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "bwandb" in default_backend:
            import wandb
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger["wandb"] = wandb

            wandb.define_metric("val-core/*", step_metric="val_step")
            wandb.define_metric("val-aux/*", step_metric="val_step")
        elif "tracking" in default_backend or "wandb" in default_backend:
            from wandb.apis.public import Api

            import wandb

            # 获取上一个运行的ID
            api = Api()
            runs = api.runs(f"skpig/{project_name}")
            if runs:
                for run in reversed(runs):
                    if run.name == experiment_name:
                        last_run_id = run.id # 最后一个运行的ID
                        resume_id = last_run_id
                        # if input(f"是否继续上次的运行？{runs[-1].url}(y/n)") == "n":
                        #     resume_id = None
                        print("继续上次的wandb运行")
                        break
                else:
                    resume_id = None
            else:
                resume_id = None

            if resume_id is None or resume_step == 0:
                run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                )
            else:
                run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    # resume_from=f"{resume_id}?_step={resume_step}"
                    resume='allow',
                    id=resume_id
                )
                print(f"Resuming wandb run {resume_id} from step {resume_step}")
            run.mark_preempting()
            self.logger["wandb"] = wandb

            wandb.define_metric("val-core/*", step_metric="val_step")
            wandb.define_metric("val-aux/*", step_metric="val_step")

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger or "bwandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()

        if "clearnml" in self.logger:
            self.logger["clearnml"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.mark_completed()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:

    def log(self, loggers, data: DataProto, inputs, outputs, tag, step):
        if 'wandb' in loggers:
            self.log_generations_to_wandb(tag, data, inputs, outputs, step)
        if 'bwandb' in loggers:
            self.log_generations_to_bwandb(tag, data, inputs, outputs, step)
        # if 'swanlab' in loggers:
        #     self.log_generations_to_swanlab(samples, step)
        # if "mlflow" in loggers:
        #     self.log_generations_to_mlflow(samples, step)

    def log_generations_to_bwandb(self, tag, data: DataProto, inputs, outputs, step):


        selected_keys_for_token_level_metrics = [
            'token_level_scores',
            'token_level_rewards',
            'advantages',
            'old_log_probs',
        ]

        raise NotImplementedError


    def log_generations_to_wandb(self, tag, data: DataProto, inputs, outputs, step):
        """Log samples to wandb as a table
        Args:
            tag (str): tag to identify the table
            samples (List[Tuple[str, str, float]]): list of samples, each sample is a tuple of (input, output and score)
            step (int): step to log the data
        """
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ['id', 'input', 'output', 'score']

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=[])

        # Add new samples to the table
        for i in range(len(inputs)):
            input_text = inputs[i]
            output_text = outputs[i]
            score = data.batch["token_level_scores"][i].sum().item()
            new_table.add_data(i, input_text, output_text, score)

        # Update reference and log
        wandb.log({f"{tag}/generations": new_table}, step=step)
        setattr(self, tag, new_table)

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
