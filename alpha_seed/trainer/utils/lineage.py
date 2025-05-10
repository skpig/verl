import logging
import re
from enum import Enum
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any, Callable, List, Optional, Union

from omegaconf import OmegaConf

_SOURCE = "alpha-seed"

logger = logging.getLogger("lineage")
logger.setLevel(logging.INFO)


class _SDKVersion(str, Enum):
    RPC_CLIENT = "rpc_client"
    HTTP_CLIENT = "http_client"
    HTTP_CLIENT_WITH_LINEAGE_BUILDER = "http_client_with_lineage_builder"
    HTTP_CLIENT_WITH_LINEAGE_AND_PROFILING_BUILDER = ("http_client_with_lineage_and_profiling_builder")


def _detect_sdk_version() -> Optional[_SDKVersion]:
    spec = find_spec("bytedmerlin")
    if spec is None:
        return None

    v = version("bytedmerlin")

    if v < "0.0.5.25":
        return _SDKVersion.RPC_CLIENT

    lineage_builder_spec = find_spec("bytedmerlin.ai_assets.report_builder")
    asset_builder_spec = find_spec("bytedmerlin.assets.reporter")

    if lineage_builder_spec is None and asset_builder_spec is None:
        return _SDKVersion.HTTP_CLIENT
    elif lineage_builder_spec is not None and asset_builder_spec is None:
        return _SDKVersion.HTTP_CLIENT_WITH_LINEAGE_BUILDER

    return _SDKVersion.HTTP_CLIENT_WITH_LINEAGE_AND_PROFILING_BUILDER


def safely_do(f: Callable, rank: int):
    """
    A decorator function that ensures the given function `f` is executed safely for lineage reporting.
    It checks the SDK version and logs warnings if the SDK is not installed or outdated.
    Also, it only executes the function on the specified rank (default is rank 0).

    Args:
        f (Callable): The function to be executed safely.
        rank (int, optional): The rank on which the function was executed.

    Returns:
        Callable: A wrapped function that can be called to execute `f` safely.
    """

    def wrapped():
        if rank != 0:
            return

        try:
            v = _detect_sdk_version()
            if v is None:
                logger.warning(
                    "bytedmerlin is not installed, aborted lineage report; please install the latest version or use the image `data.aml.verl` with version v190 or higher",
                    stacklevel=2,
                )
                return

            if v == _SDKVersion.RPC_CLIENT:
                logger.warning(
                    "old rpc bytedmerlin is detected; please upgrade your bytedmerlin to the latest version or use the image `data.aml.verl` with version v190 or higher",
                    stacklevel=2,
                )

            ret = f()
            logger.info(f"lineage reported: {ret}", stacklevel=2)
            return ret
        except Exception as e:
            logger.warning(f"error is raised when trying to report lineage safely: {e}", stacklevel=2)

    return wrapped


def report_job_config(config: Any):
    """
    Report the job configuration.
    This function converts the input configuration object to a Python container using OmegaConf,
    then uses the MerlinJobReportBuilder to report the job configuration asynchronously.

    Args:
        config (Any): The job configuration object, which can be of any type supported by OmegaConf.

    Returns:
        The result of the report method from MerlinJobReportBuilder, which typically contains
        information about the reporting operation.
    """
    from bytedmerlin.ai_assets.report_builder import MerlinJobReportBuilder

    config = OmegaConf.to_container(config)

    return MerlinJobReportBuilder().report(_SOURCE, is_async=True, job_config=config)


_MAGNUS_EXPR_PATTERN = r"(magnus:\/\/)?(\w+)\.(\w+)\.(\w+)(.*)"


def _is_magnus_expr(s: str):
    return re.match(_MAGNUS_EXPR_PATTERN, s) is not None


def report_data_loaded(train_files: Union[str, List[str]], val_files: Union[str, List[str]]):
    """
    Report the training and validation data files that have been loaded.
    It differentiates between Magnus expressions and regular HDFS paths, and reports them accordingly.

    Args:
        train_files (Union[str, List[str]]): The path or list of paths to the training data files.
        val_files (Union[str, List[str]]): The path or list of paths to the validation data files.

    Returns:
        The result of the report method from MerlinJobReportBuilder.
    """
    from bytedmerlin.ai_assets.report_builder import (
        MerlinJobReportBuilder,
        hdfs_dataset,
    )
    from bytedmerlin.ai_assets.reporter_ids import EVENT_TYPE_TRIAL_DATA_LOAD
    from bytedmerlin.ai_assets.utils import get_region

    if isinstance(train_files, str):
        train_files = [train_files]

    if isinstance(val_files, str):
        val_files = [val_files]

    builder = MerlinJobReportBuilder()

    for path in train_files:
        if _is_magnus_expr(path):
            # If the path is a Magnus expression, report it as an Iceberg dataset
            builder = builder.add_inputs({
                "category": "dataset",
                "subcategory": "iceberg",
                "region": get_region(),
                "tag": "data.train_path",
                "magnus_expr": path,  # reporting service will infer table name and snapshot id from this
            })
        else:
            # If the path is a regular HDFS path, report it using hdfs_dataset
            builder = builder.add_inputs(hdfs_dataset(hdfs_path=path, tag="data.train_path"))

    for path in val_files:
        if _is_magnus_expr(path):
            builder = builder.add_inputs({
                "category": "dataset",
                "subcategory": "iceberg",
                "region": get_region(),
                "tag": "data.val_path",
                "magnus_expr": path,  # reporting service will infer table name and snapshot id from this
            })
        else:
            builder = builder.add_inputs(hdfs_dataset(hdfs_path=path, tag="data.val_path"))

    return builder.report(_SOURCE, is_async=True, event_type=EVENT_TYPE_TRIAL_DATA_LOAD)


def report_trial_started(
    checkpoint_paths: List[str],
):
    """
    Report the start of a trial along with the checkpoint paths used.
    It adds each checkpoint path in the provided list as an input asset to the report.

    Args:
        checkpoint_paths (List[str]): A list of paths to the checkpoints used at the start of the trial.

    Returns:
        The result of the report method from MerlinJobReportBuilder.
    """
    from bytedmerlin.ai_assets.report_builder import (
        MerlinJobReportBuilder,)
    from bytedmerlin.ai_assets.reporter_ids import EVENT_TYPE_TRIAL_STARTED
    from bytedmerlin.ai_assets.utils import get_region

    builder = MerlinJobReportBuilder()
    region = get_region()

    for path in checkpoint_paths:
        builder = builder.add_inputs({
            "region": region,
            "category": "model",
            "subcategory": "hdfs_ckpt",
            "checkpoint_path": path,
        })

    return builder.report(_SOURCE, is_async=True, event_type=EVENT_TYPE_TRIAL_STARTED)


def report_trial_ckpts_load(checkpoint_infos: dict):
    """
    Report the checkpoints loaded at the start of a trial.
    
    Args:
        checkpoint_infos (dict): A dictionary containing checkpoint information.
            Keys are checkpoint paths, and values are dictionaries with additional
            information such as 'step', 'epoch', 'extra', and 'tag'.

    Returns:
        The result of the report method from MerlinJobReportBuilder.
    """
    from bytedmerlin.ai_assets.report_builder import MerlinJobReportBuilder
    from bytedmerlin.ai_assets.reporter_ids import EVENT_TYPE_TRIAL_STARTED
    from bytedmerlin.ai_assets.utils import get_region

    builder = MerlinJobReportBuilder()
    region = get_region()

    for path, ckpt_info in checkpoint_infos.items():
        # Create base input asset
        input_asset = {
            "region": region,
            "category": "model",
            "subcategory": "hdfs_ckpt",
            "checkpoint_path": path,
            **{
                k: v for k, v in ckpt_info.items() if k in ["step", "epoch", "extra", "tag"]
            }
        }

        builder = builder.add_inputs(input_asset)

    return builder.report(_SOURCE, is_async=True, event_type=EVENT_TYPE_TRIAL_STARTED)


def report_rl_ckpts_load(worker_configs: dict):
    """
    Report the RL trianing special worker checkpoints loaded at the start of a trial.
    This function iterates over the provided worker configurations and reports the checkpoint paths
    for each role as input assets.

    Args:
        worker_configs (dict): A dictionary containing configuration information for each role.
            Keys are role names, and values are dictionaries with model configuration details.

    Returns:
        The result of the report method from MerlinJobReportBuilder
    """
    from bytedmerlin.ai_assets.report_builder import (
        MerlinJobReportBuilder,)
    from bytedmerlin.ai_assets.reporter_ids import EVENT_TYPE_TRIAL_STARTED
    from bytedmerlin.ai_assets.utils import get_region

    builder = MerlinJobReportBuilder()
    region = get_region()

    for role, model_config in worker_configs.items():
        try:
            path = model_config.get("model", {}).get("path", "")
            if not path:
                logger.info(f"Model {role} path is empty, skip lineage")
                continue
        except Exception as e:
            logger.info(
                f"Unable to get lineage info for role {role}, skip lineage. No affect to training process. {str(e)}")
            continue

        input_asset = {
            "region": region,
            "category": "model",
            "subcategory": "hdfs_ckpt",
            "checkpoint_path": path,
            "tag": role
        }

        builder = builder.add_inputs(input_asset)

    return builder.report(_SOURCE, is_async=True, event_type=EVENT_TYPE_TRIAL_STARTED)


def report_checkpoint_saved(
    path: str,
    step: int,
    epoch: int = -1,
    default_hdfs_path: Optional[str] = None,
    **kwargs,
):
    """
    Report that a checkpoint has been saved. This function uses the MerlinJobReportBuilder
    to report the saved checkpoint information asynchronously.

    Args:
        path (str): The actual HDFS path where the checkpoint is saved.
        step (int): The training step at which the checkpoint was saved.
        epoch (int, optional): The training epoch at which the checkpoint was saved. Defaults to -1.
        default_hdfs_path (str): The default HDFS path associated with the checkpoint.
        **kwargs: Additional keyword arguments to be passed to the hdfs_checkpoint function.

    Returns:
        The result of the report method from MerlinJobReportBuilder.
    """
    from bytedmerlin.ai_assets.report_builder import (
        MerlinJobReportBuilder,
        hdfs_checkpoint,
    )
    from bytedmerlin.ai_assets.reporter_ids import EVENT_TYPE_CKPT_SAVED

    builder = MerlinJobReportBuilder()
    extra = kwargs.pop("extra", {})
    extra = update_extra_with_robust_info(extra)

    if default_hdfs_path is None:
        checkpoint_index = path.find("/checkpoints/")
        if checkpoint_index != -1:
            extra["default_hdfs_path"] = path[:checkpoint_index]

    builder = builder.add_outputs(hdfs_checkpoint(
        hdfs_path=path,
        step=step,
        epoch=epoch,
        extra=extra,
        **kwargs,
    ))

    return builder.report(_SOURCE, is_async=True, event_type=EVENT_TYPE_CKPT_SAVED)


def update_extra_with_robust_info(extra: dict):
    extra = {}
    from os import environ
    robust_run_id = environ.get("ROBUST_RUN_ID", None)
    if robust_run_id is not None:
        extra["robust_run_id"] = robust_run_id
    return extra
