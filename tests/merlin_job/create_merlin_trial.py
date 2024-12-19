import argparse
import copy
import json
import logging
import numbers
import os
import pprint
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from string import Template
from time import localtime, strftime
from typing import Any, Dict, List, Optional, Union

import requests
import statistics_function
import yaml
from bytedrh2.http_client import authenticate
from utils import (create_rh2_client, get_arnold_trial_info, get_at_user_msg, get_job_run_info,
                   get_run_id_from_exp_name, get_run_id_from_job_info, get_trial_id_from_job_info, launch_job, query,
                   query_tracking_running_steps, stop_job, version_to_image_url)

mute_send_msg_to_lark = os.environ.get('ALPHASEED_NIGHTLY_CI_MUTE_SEND_MESSAGE_TO_LARK', 'yes') == 'yes'

lark_msg_template = Template("""Daily task launched:
Task Name: $TASK_NAME
Date: $DATE
CI actor: $CI_ACTOR
Config Path: $CONFIG_PATH
Merlin Trial: $MERLIN_TRIAL
""")

GLOBAL_JOB_ID = None
GLOBAL_RH2_CLIENT = None
GLOBAL_TRIAL_STAUTS = None


def send_message_to_alphaseed_ci_group(msg):
    if mute_send_msg_to_lark:
        print('[debug] mute send message to lark')
        print(msg)
    else:
        # corresponding to lark group "AlphaSeed 每日基准任务测试结果"; ID: 7446275229449224194
        url = "https://open.larkoffice.com/open-apis/bot/v2/hook/20c7bb45-3a9a-491c-95cc-24ad37a7f1f5"

        req = {"msg_type": "text", "content": {"text": msg}}
        payload = json.dumps(req)
        response = requests.request("POST", url, data=payload)
        print(response.content)  # Print Response


def get_latest_scm_production_version(repo, last_num_only=True):
    """Check the latest scm production version by actually cloning it.
    Make sure the size of the repo is not large.
    repo is something like aml/mlsys/megatron
    """
    repo_name = '%2F'.join(repo.split('/'))
    # TODO(zhangchi.usc1992) retry to handle network failure
    scm_version = subprocess.check_output(
        args=f"curl --location 'scm.byted.org/api/v2/versions/newest_version/?repo_name={repo_name}&type=online&arch=x86_64&need_sync_bvc=0'",  # noqa E501
        shell=True,
    ).decode()[1:-1]
    if last_num_only:
        scm_version = scm_version.split('.')[-1]
    return scm_version


def get_scm_production_repo_id(repo):
    """Check the scm repo id
    repo is something like aml/mlsys/janus_dev
    """
    repo_name = '%2F'.join(repo.split('/'))
    # TODO(luqi.6789) retry to handle network failure
    scm_info_str = subprocess.check_output(
        args=f"curl --location 'scm.byted.org/api/v2/repos/by_names/?repo_names={repo_name}&type=online&arch=x86_64&need_sync_bvc=0'",  # noqa E501
        shell=True,
    ).decode()[1:-1]
    scm_info: Dict = json.loads(scm_info_str, strict=True)
    return scm_info.get('id', None)


def get_latest_commit_sha(repo, branch_name='master'):
    result = subprocess.check_output(f"git ls-remote {repo} refs/heads/{branch_name} | awk '{{ print $1}}'",
                                     shell=True).decode()[:-1]
    return result


def get_full_scm_production_name_from_template(production_prefix_name: str, scms_in_template) -> str:
    full_scm_production_name = None
    for i, scm_repo in enumerate(scms_in_template):
        if production_prefix_name in scm_repo['name']:
            full_scm_production_name = scm_repo['name']

    scm_names = [scm_repo['name'] for scm_repo in scms_in_template]
    assert (full_scm_production_name
            is not None), f"Expected production_prefix_name({production_prefix_name}) must in scms({scm_names})"
    return full_scm_production_name


def get_all_metric_running_steps_from_tracking(project_id, run_id, metrics):
    running_steps = [query_tracking_running_steps(project_id, run_id, metric) for metric in metrics]
    return running_steps


def get_run_id(project_id, experiment_name, args, baseline_project_id=None):
    try:
        run_id = get_run_id_from_exp_name(project_id=project_id, experiment_name=experiment_name)
    except Exception as e:
        logging.warning(
            f"get_run_id_from_exp_name failed. Use args.run_id instead. Exception as below: {str(e)}"  # noqa E501
        )
        run_id = None

    return run_id, project_id


@dataclass
class ComparisonResult:
    success: List[bool]
    diff_results: Union[Dict[int, numbers.Number], numbers.Number]
    intersection_steps: Optional[List[int]] = None
    interval_baseline_value: Optional[numbers.Number] = None
    interval_running_value: Optional[numbers.Number] = None

    def __str__(self):
        return f"comparison success: {self.success}, diff_results: {self.diff_results}"


class MetricsStatisticsProcessor:

    # statics_function_module: str = 'statistics_function'
    conclusion_level: Dict[int, str] = {0: 'Result matches', 1: 'Result mismatches. Requires further manual check'}

    def __init__(self, metric_name: str, metric_config: Dict[str, Any]):
        self.metric_name = metric_name
        self.metric_config = copy.deepcopy(metric_config)
        self.metrics_statistics_func_config = None
        self.conclusion = 0
        self.statistics_results = None
        self.analysis_str: List[str] = []

        self._parse_metric_config()

    def _parse_metric_config(self):
        self.align_mode = self.metric_config.get('align_mode', 'step_by_step')
        assert self.align_mode in ['step_by_step', 'interval']
        self.align_level = self.metric_config.get('align_level', 'bitwise')

        if self.align_level == 'bitwise':
            assert (self.align_mode == "step_by_step"
                   ), f"align_mode must be step_by_step in bitwise mode, but gets metric_config = {self.metric_config}"
            assert (
                "statistics_function" not in self.metric_config
            ), f"statistics_function is not allowed in bitwise mode, but gets metric_config = {self.metric_config}"
            self.metrics_statistics_func_config = [{
                "name": "absolute_diff",
                "tolerance_lower_bound": 0,
                "tolerance_upper_bound": 0
            }]

        elif self.align_level == 'tolerance':
            assert (
                "statistics_function" in self.metric_config
            ), f"statistics_function is required in tolerance mode, but gets metric_config = {self.metric_config}"
            self.metrics_statistics_func_config = self.metric_config['statistics_function']
            if not isinstance(self.metrics_statistics_func_config, list):
                self.metrics_statistics_func_config = [self.metrics_statistics_func_config]
        else:
            raise ValueError(f"align_level must be bitwise or tolerance, but got {self.align_level}")

    @property
    def moving_average_function(self):
        assert self.align_mode == 'step_by_step', f"align_mode must be step_by_step, but gets {self.align_mode}"
        if "moving_average_window" not in self.metric_config:
            logging.warning("'moving_average_window' not in self.metric_config. Returns None")
            return None
        window_size = self.metric_config['moving_average_window']
        func = self._import_statistics_function("simple_moving_average")
        func = partial(func, window_size=window_size)
        return func

    def _import_statistics_function(self, func_name: str):
        func = getattr(statistics_function, func_name, None)
        return func

    def process(self, project_name, run_id, baseline_run_id):
        """
        process the metric statistics
        """
        # get baseline results
        baseline_results = query(project_name, baseline_run_id, self.metric_name)
        # get running results
        running_results = query(project_name, run_id, self.metric_name)

        self.statistics_results = self._calculate_diff(baseline_results, running_results)
        self.conclusion = self._collect()

        return self.conclusion

    def _calculate_diff(self, baseline_results: Dict[Any, Any], running_results: Dict[Any, Any]):
        """
        calculate diff between baseline and running results
        """
        self.baseline_vals = []
        self.running_vals = []
        self.intersection_steps = []

        # find intersection of baseline and running results
        for step, baseline_val in baseline_results.items():
            if not isinstance(baseline_val, numbers.Number):
                print(f'baseline_val must be a number, Got {baseline_val}')
                continue

            if step in running_results:
                running_val = running_results[step]

                if not isinstance(running_val, numbers.Number):
                    print(f'val must be a number, Got {running_val}')
                    continue

                self.intersection_steps.append(step)
                self.baseline_vals.append(baseline_val)
                self.running_vals.append(running_val)

        self.num_overlap = len(self.baseline_vals)
        statistics_results: Dict[str, ComparisonResult] = {}

        if self.align_mode == 'step_by_step':
            average_function = self.moving_average_function
            if average_function is not None:
                self.baseline_vals, self.running_vals = average_function(self.baseline_vals, self.running_vals)

            for statistics_func_config in self.metrics_statistics_func_config:
                func_name = statistics_func_config.get('name', None)
                compare_func = self._import_statistics_function(func_name)
                comparison_result = ComparisonResult(success=[], diff_results={}, intersection_steps=[])
                if compare_func is None:
                    raise ValueError(f'statistics_function {func_name} not found')
                for step, baseline_val, running_val in zip(self.intersection_steps, self.baseline_vals,
                                                           self.running_vals):
                    success, diff_result, _, _ = compare_func(baseline_val, running_val, **statistics_func_config)
                    comparison_result.success.append(success)
                    comparison_result.diff_results[step] = diff_result
                    comparison_result.intersection_steps.append(step)

                statistics_results[func_name] = comparison_result

        elif self.align_mode == 'interval':
            for statistics_func_config in self.metrics_statistics_func_config:
                func_name = statistics_func_config.get('name', None)
                compare_func = self._import_statistics_function(func_name)
                comparison_result = ComparisonResult(success=[],
                                                     diff_results=0,
                                                     intersection_steps=self.intersection_steps)
                success, diff_result, interval_baseline_val, interval_running_val = compare_func(
                    self.baseline_vals, self.running_vals, **statistics_func_config)
                comparison_result.success = [success]
                comparison_result.diff_results = diff_result
                comparison_result.interval_baseline_value = interval_baseline_val
                comparison_result.interval_running_value = interval_running_val
                statistics_results[func_name] = comparison_result
        else:
            raise NotImplementedError(f'align_mode must be step_by_step or interval, but got {self.align_mode}')

        return statistics_results

    def _collect(self):
        """
        collect the diff results and generate conclusion
        """
        if self.num_overlap == 0:
            self._add_analysis_line(f"metric: {self.metric_name}: can't find any overlap step. Please manual check!")
            return 1

        assert len(self.statistics_results) > 0, "statistics_results is empty, please check"

        assert self.align_level in [
            'bitwise',
            'tolerance',
        ], f"align_level must be bitwise or tolerance, but got {self.align_level}"
        conclusion = 0
        for statistics_name, comparison_result in self.statistics_results.items():
            if all(comparison_result.success):
                if self.align_mode == 'step_by_step':
                    self._add_analysis_line(f"  {statistics_name}: can align with {self.align_level}.")
                else:
                    self._add_analysis_line(
                        f"  {statistics_name}: can align with {self.align_level} (baseline val = {comparison_result.interval_baseline_value:.3f}, current val = {comparison_result.interval_running_value:.3f})."  # noqa: E501
                    )
            else:
                conclusion = 1
                if self.align_mode == 'interval':
                    self._add_analysis_line(f"  {statistics_name} can't align with {self.align_level}.")
                    assert isinstance(
                        comparison_result.diff_results, numbers.Number
                    ), f"diff_results must be a number in interval mode, but got {comparison_result.diff_results}"
                    self._add_analysis_line(
                        f"  Interval: {min(comparison_result.intersection_steps)}-{max(comparison_result.intersection_steps)}, diff: {comparison_result.diff_results:.3f}"  # noqa: E501
                        f"(baseline val = {comparison_result.interval_baseline_value:.3f}, current val = {comparison_result.interval_running_value:.3f})."  # noqa: E501
                    )
                elif self.align_mode == 'step_by_step':
                    max_show_diff_steps = 5
                    assert isinstance(
                        comparison_result.diff_results, dict
                    ), f"diff_results must be a dict in step_by_step mode, but got {comparison_result.diff_results}"
                    self._add_analysis_line(
                        f"  {statistics_name} can\'t align with {self.align_level}. Only show {max_show_diff_steps} steps with {statistics_name} as below:"  # noqa: E501
                    )

                    failed_results: Dict[int,
                                         numbers.Number] = copy.deepcopy(comparison_result.diff_results)  # <step, diff>
                    # only keep the failed results
                    for step in failed_results.copy():
                        if comparison_result.success[self.intersection_steps.index(step)]:
                            failed_results.pop(step)

                    # get max abs diff from failed results
                    top_steps_with_abs_diff = sorted(failed_results,
                                                     key=lambda step: abs(failed_results.get(step)),
                                                     reverse=True)
                    mismatch_ratio = len(top_steps_with_abs_diff) / len(comparison_result.diff_results)
                    self._add_analysis_line(
                        f"  (Mismacthes ratio: {mismatch_ratio * 100:.2f}%, where {len(top_steps_with_abs_diff)} steps mismatches among {len(comparison_result.diff_results)} steps)"  # noqa: E501
                    )

                    top_steps_with_abs_diff = top_steps_with_abs_diff[:max_show_diff_steps]
                    for step in top_steps_with_abs_diff:
                        baseline_val = self.baseline_vals[self.intersection_steps.index(step)]
                        running_val = self.running_vals[self.intersection_steps.index(step)]
                        diff_result = failed_results[step]
                        self._add_analysis_line(
                            f"      Step: {step}, baseline: {baseline_val:.3f}, running: {running_val:.3f}, diff: {diff_result:.3f}"  # noqa: E501
                        )
                else:
                    raise NotImplementedError(f'align_mode must be step_by_step or interval, but got {self.align_mode}')

        return conclusion

    def get_conclusion(self):
        return self.conclusion

    def _add_analysis_line(self, line: str):
        self.analysis_str.append(line)

    def print_analysis_str(self):
        # insert head of result
        msg = self.get_analysis_str()
        print(msg)

    def get_analysis_str(self):
        summary_one_line = f"[metric] {self.metric_name}: {self.conclusion_level[self.conclusion]}\n"
        lines = [summary_one_line]
        for line in self.analysis_str:
            lines.append(line + "\n")
        lines.append('\n')
        return ''.join(lines)


@dataclass
class SCMProductionInfo:
    production_name: str
    version: str
    repo_id: int
    link: Optional[str] = None

    def __post_init__(self):
        if not self.version:
            self.link = ""
        else:
            self.link = f'{self.production_name} scm (https://cloud.bytedance.net/scm/detail/{self.repo_id}/versions): {self.version}\n'  # noqa: E501


class SCMManager:
    """
    Read scm config from config_path, and update scm version and env in job_def_version
    """

    override_env_registry: Dict[str, str] = {}

    def __init__(self, config_path: str, filter: Optional[Union[List[str], str]] = None):
        self.config_path = config_path
        self.user_config = None
        self.filter = filter
        if self.filter is not None:
            self.filter = filter if isinstance(filter, list) else [filter]

        self._parse_config()

    def _parse_config(self):
        with open(self.config_path) as f:
            config = json.load(f)

        if self.filter is None:
            self.user_config = config
        else:
            self.user_config = {key: config[key] for key in self.filter if key in config}

    def _update_env(self, job_def_version: Dict[str, Any]):
        for user_config_scm_name, user_scm_config in self.user_config.items():
            if user_config_scm_name not in SCMManager.override_env_registry:
                # skip if need to update scm repo info, but not update env
                continue

            user_config_version = user_scm_config.get('version', "")
            # get latest version of scm
            if user_config_version == 'None' or user_config_version == "":
                user_config_version = get_latest_scm_production_version(user_config_scm_name, last_num_only=False)

            self.user_config[user_config_scm_name]['version'] = user_config_version
            # update env
            func_name = SCMManager.override_env_registry[user_config_scm_name]
            getattr(self, func_name)(job_def_version, user_config_version)

    def _update_scm(self, job_def_version: Dict[str, Any]):
        for user_config_scm_name, user_scm_config in self.user_config.items():
            # if user_config_scm_name in SCMManager.override_env_registry:
            #     # skip if need to update env, but not update scm repo info
            #     continue

            user_config_version = user_scm_config.get('version', "")
            # get latest version of scm
            if user_config_version == 'None' or user_config_version == "":
                user_config_version = get_latest_scm_production_version(user_config_scm_name, last_num_only=False)

            append_new_scm = True
            for i, scm_repo in enumerate(job_def_version['scms']):
                if user_config_scm_name == scm_repo['name']:
                    # update exists config
                    append_new_scm = False
                    job_def_version['scms'][i]['version'] = user_config_version
                    job_def_version['scms'][i]['mnt'] = user_scm_config['mnt']
                    break

            if append_new_scm:
                job_def_version['scms'].append({
                    "name": user_config_scm_name,
                    "version": user_config_version,
                    "mnt": user_scm_config['mnt'],
                    "pypath": [],
                })

            self.user_config[user_config_scm_name]['version'] = user_config_version

    def update_job_def_version(self, job_def_version: Dict[str, Any]):
        self._update_env(job_def_version)
        self._update_scm(job_def_version)
        return job_def_version


class PipManager:
    """
    Read pip config from config_path, and update pip version in job_def_version
    """

    def __init__(self, config_path: str, filter: Optional[Union[List[str], str]] = None):
        self.config_path = config_path
        self.user_config = None
        self.filter = filter
        if self.filter is not None:
            self.filter = filter if isinstance(filter, list) else [filter]
        self._parse_config()

    def _parse_config(self):
        with open(self.config_path) as f:
            txt_config = f.readlines()

        self.user_config = []
        for line in txt_config:
            line = line.strip()
            # e.g.: 'byted-mason==0.1.2'

            package_name = line.split('==')[0] if '==' in line else line
            if self.filter is None or package_name in self.filter:
                self.user_config.append(line)

    def update_job_def_version(self, job_def_version: Dict[str, Any]):
        job_def_version['pip3'] = self.user_config
        return job_def_version


def main():
    # This script can be run in two ways. manual or cron.
    # in manual form, user has to input some variables and will be stored as environment variable
    # in cron form, all the infos are automatically extracted and stored as environment variable

    with open('baselines/arnold_group.json') as f:
        all_groups = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_folder', required=True, type=str)
    parser.add_argument('--meta_info_config_name', required=True, type=str)
    parser.add_argument('--metrics_config_name', required=True, type=str)
    parser.add_argument('--base_image_url', default=None)
    parser.add_argument('--base_image_version', default=None)
    parser.add_argument('--alphaseed_git_branch', type=str, default='master', help='will use master by default')
    parser.add_argument('--alphaseed_git_commit', type=str, default=None, help='will use the latest commit by default')
    parser.add_argument('--experiment_name',
                        type=str,
                        default=None,
                        help='experiment_name. Will use original experiment name + date')
    parser.add_argument(
        '--owner_name',
        type=str,
        default='',
        help='experiment owner name. Will @ corresponding author of the experiment in lark message.',
    )
    parser.add_argument('--wait_result', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--start_trial', type=str, default='True', choices=['True', 'False'])  # for debugging purpose
    parser.add_argument('--group', type=str, default=None)

    args = parser.parse_args()

    pprint.pprint(vars(args))

    config_folder = args.config_folder
    meta_info_config_name = args.meta_info_config_name
    metrics_config_name = args.metrics_config_name
    base_image_url = args.base_image_url
    base_image_version = args.base_image_version
    alphaseed_git_branch = args.alphaseed_git_branch
    alphaseed_git_commit = args.alphaseed_git_commit
    experiment_name = args.experiment_name
    owner_name = args.owner_name

    wait_result = eval(args.wait_result.capitalize())
    start_trial = eval(args.start_trial.capitalize())
    group = args.group

    # override local variable with environment variable. It's ugly, but works
    def override_local_with_env(env_name, variable):
        if env_name in os.environ and len(os.getenv(env_name)) > 0:
            return os.getenv(env_name)
        else:
            return variable

    alphaseed_git_branch = override_local_with_env('ALPHASEED_BRNACH_NAME', alphaseed_git_branch)
    alphaseed_git_commit = override_local_with_env('ALPHASEED_COMMIT_SHA', alphaseed_git_commit)
    owner_name = override_local_with_env('ALPHASEED_NIGHTLY_CI_EXP_OWNER_NAME', owner_name)

    base_image_version = override_local_with_env('ALPHASEED_BASE_IMAGE_VERSION', base_image_version)
    if base_image_version == 'other':
        base_image_url = override_local_with_env('ALPHASEED_BASE_IMAGE_URL', base_image_url)
        assert base_image_url is not None, 'base_image_url is not provided'
    else:
        base_image_url = version_to_image_url(base_image_version)

    group = override_local_with_env('ALPHASEED_GROUP', group)

    with open(os.path.join(config_folder, meta_info_config_name)) as f:
        meta_info = json.load(f)

    baseline_trial_url = meta_info['baseline_trial_url']
    baseline_run_id = meta_info['baseline_run_id']

    #  meta_info['pip_requirements'] 不能指定版本号，用于判断所需的package 类别用
    # pip_requirements 是用户提供的，需要用户自己指定版本号
    # template.json 的提供无效，会被覆盖
    pip_requirements = meta_info['pip_requirements']
    scm_requirements = meta_info['scm_requirements']

    running_steps = meta_info['running_steps']

    with open(os.path.join(config_folder, 'entrypoint_template')) as f:
        entrypoint_template = f.read()

    with open(os.path.join(config_folder, 'template.json')) as f:
        template = json.load(f)

    # configurations of metrics and corresponding diff function for
    # comparing the diff between baseline trial and current trial
    metrics_config_file = os.path.join(config_folder, metrics_config_name)

    with open(metrics_config_file) as f:
        metrics_dict = json.load(f)

    metric_names = metrics_dict.keys()

    # initialize metrics statistics processor. Each metric corresponds to a processor
    metric_processors: List[MetricsStatisticsProcessor] = []
    for metric_name in metrics_dict:
        metric_configuration = metrics_dict[metric_name]
        metric_processors.append(MetricsStatisticsProcessor(metric_name=metric_name,
                                                            metric_config=metric_configuration))

    # update default arguments
    if alphaseed_git_commit is None:
        alphaseed_git_commit = get_latest_commit_sha('git@code.byted.org:seed/alpha-seed.git',
                                                     branch_name=alphaseed_git_branch)
        print(f"alphaseed_git_commit = {alphaseed_git_commit}")
    if experiment_name is None:
        current_time_str = strftime("%m-%d-%Y-%H-%M-%S", localtime())
        experiment_name = meta_info['default_experiment_name'] + '_' + current_time_str

    # populate default arguments into template
    template['jobDefVersion']['gitRepo']['branchName'] = alphaseed_git_branch
    template['jobDefVersion']['gitRepo']['commitSha'] = alphaseed_git_commit
    template['jobDefVersion']['imageMeta']['imageUrl'] = base_image_url

    if group is not None:
        group_id = all_groups[group]['groupIds']
        cluster_id = all_groups[group]['clusterId']
        template['jobRunParams']['resource']['arnoldConfig']['groupIds'] = group_id
        template['jobRunParams']['resource']['arnoldConfig']['clusterId'] = cluster_id

        print(f"using group_id = {group_id}")
        print(f"using cluster_id = {cluster_id}")

    # update entrypoint
    entrypoint_replace_name = {'EXPERIMENT_NAME': experiment_name, 'PROJECT_NAME': meta_info['project']}

    entrypoint_template = Template(entrypoint_template)
    entrypoint_template = entrypoint_template.safe_substitute(entrypoint_replace_name)

    # update template
    template['jobRunParams']['entrypointFullScript'] = entrypoint_template

    # update env if file env.yaml exists
    env_file_path = os.path.join(config_folder, 'env.yaml')
    if os.path.exists(env_file_path):
        with open(env_file_path) as f:
            env = yaml.load(f, Loader=yaml.BaseLoader)

    # read scm version and mnt info from `scm_config`, and update job_def_version
    scm_manager = SCMManager(os.path.join(config_folder, 'scm_requirements.json'), filter=scm_requirements)
    scm_manager.update_job_def_version(template['jobDefVersion'])

    # read pip packages version from `pip_config`, and update job_def_version
    pip_manager = PipManager(os.path.join(config_folder, 'pip_requirements.txt'), filter=pip_requirements)
    pip_manager.update_job_def_version(template['jobDefVersion'])

    print("======= template =======")
    print(template)

    if start_trial:
        # get token via environment variable as the owner of the MR
        username = os.environ["CI_ACTOR"]
        token_cn = os.environ["MERLIN_CN_TOKEN_" + username.upper().replace(".", "_")]

        authenticate(host='rh2.bytedance.net', user_name=username, access_token=token_cn)

        # launch job
        rh2_client = create_rh2_client()

        global GLOBAL_RH2_CLIENT
        GLOBAL_RH2_CLIENT = rh2_client

        job_id = launch_job(rh2_client=rh2_client, task_config=template)

        global GLOBAL_JOB_ID
        GLOBAL_JOB_ID = job_id

        job_info = get_job_run_info(rh2_client=rh2_client, job_id=job_id)

        # send a lark message
        merlin_trial_id = f'https://ml.bytedance.net/development/instance/jobs/{job_id}'
        lark_message_kwargs = dict(
            DATE=strftime("%Y-%m-%d %H:%M:%S", localtime()),
            CI_ACTOR=username,
            CONFIG_PATH=config_folder,
            MERLIN_TRIAL=merlin_trial_id,
            EXP_OWNER=owner_name,
            TASK_NAME=meta_info['default_experiment_name'],
        )
        lark_message = lark_msg_template.substitute(lark_message_kwargs)
        print(lark_message)

        send_message_to_alphaseed_ci_group(lark_message)

    if wait_result:
        queue_timeout = 1 * 3600

        baseline_project_id = meta_info['project_id']
        baseline_project_name = meta_info['project']

        while 'arnoldJobInfo' not in job_info['jobRun']['meta']:
            # There is no arnoldJobInfo at the begining of the trial
            time.sleep(5)
            job_info = get_job_run_info(rh2_client=rh2_client, job_id=job_id)

        timeout = meta_info['max_run_time (hours)'] * 3600  # in seconds
        # wait for trial to finish. There are many ways this could break. For example,
        wait_time = 120  # check every 2 minutes
        trial_id = get_trial_id_from_job_info(job_info)

        result_lark_message = ""
        result_lark_message += (
            f'AlphaSeed master commit: https://code.byted.org/seed/alpha-seed/commit/{alphaseed_git_commit}\n')
        result_lark_message += f'Config Path: {config_folder}\n'
        result_lark_message += f'Current trial: {merlin_trial_id}\n'
        result_lark_message += f'Baseline trial: {baseline_trial_url}\n'

        # wait for start
        global GLOBAL_TRIAL_STAUTS
        start_time = time.time()
        while True:
            GLOBAL_TRIAL_STAUTS = get_arnold_trial_info(trial_id)['runs'][0]['status']
            if GLOBAL_TRIAL_STAUTS not in ['disabled', 'queued', 'staging']:
                break
            logging.info(f'Current status of trial({trial_id}) is {GLOBAL_TRIAL_STAUTS}. Waiting ...')
            time.sleep(wait_time)
            if time.time() - start_time >= queue_timeout:
                break

        print(f'Current status is {GLOBAL_TRIAL_STAUTS}')

        if GLOBAL_TRIAL_STAUTS in ['queued', 'scheduled']:
            stop_job(rh2_client, job_id)
            msg_header = 'Trial still queues after 1 hours! Skip e2e ci today.\n'
            msg_header += f'Task Name: {meta_info["default_experiment_name"]} {get_at_user_msg(owner_name)}\n'
            result_lark_message = msg_header + result_lark_message
            send_message_to_alphaseed_ci_group(result_lark_message)
            sys.exit(1)

        project_id = baseline_project_id
        run_id = None

        # start running
        start_time = time.time()
        while True:
            if run_id is None:
                run_id, project_id = get_run_id(project_id,
                                                experiment_name,
                                                args,
                                                baseline_project_id=baseline_project_id)

            log_str = f"[debug] heart-beat check: start running run_id:{run_id}, required running_steps:{running_steps}, already running {time.time() - start_time} seconds"
            if run_id is not None:
                logging.info(
                    log_str +
                    f" min_metric_step:{min(get_all_metric_running_steps_from_tracking(baseline_project_name, run_id, metric_names))}"
                )
            else:
                logging.info(log_str)

            if (run_id is not None and running_steps > 0 and
                    min(get_all_metric_running_steps_from_tracking(baseline_project_name, run_id,
                                                                   metric_names)) >= running_steps):
                print("Reach running max steps. Break")
                break
            GLOBAL_TRIAL_STAUTS = get_arnold_trial_info(trial_id)['runs'][0]['status']
            if GLOBAL_TRIAL_STAUTS not in ['running']:
                print(f"status({GLOBAL_TRIAL_STAUTS}) is not running. Break")
                break
            if time.time() - start_time >= timeout:
                print("timeout. Break")
                result_lark_message += f'Warning: the trial has reached the max running time and may not reach the required steps. {get_at_user_msg(owner_name)}\n'
                break
            time.sleep(wait_time)

        if run_id is None:
            logging.warning(
                f"run_id is None. It's possible that the tracking {experiment_name} is not found in project {baseline_project_name}."  # noqa: E501
            )
            target_worker = meta_info.get('target_worker', 0)
            print(f'Target worker: {target_worker}')

            try:
                project_id, run_id = get_run_id_from_job_info(job_info, target_workers=str(target_worker))
            except Exception as e:
                print(e)
                run_id = None

        # stop the trial
        if GLOBAL_TRIAL_STAUTS not in ['failed', 'finished']:
            ret_str = stop_job(rh2_client, job_id)
            print('Stopping trial')
            print(f"stop_job ret_str:{ret_str}")

        print(f"[debug] run_id:{run_id}")

        assert (run_id
                is not None), "run_id is None. It's possible that the trial exited before the tracking was created."

        if GLOBAL_TRIAL_STAUTS in ['failed']:
            result_lark_message += f'Trial fails! Please manual check. {get_at_user_msg(owner_name)}\n'
            send_message_to_alphaseed_ci_group(result_lark_message)
            sys.exit(1)

        if GLOBAL_TRIAL_STAUTS in ['queued']:
            result_lark_message += 'Trial still queues after '

        current_tracking_link = f'https://ml.bytedance.net/experiment/tracking/detail?Id={project_id}&selectedTrial={run_id}&tab=CHART'  # noqa: E501
        baseline_tracking_link = f'https://ml.bytedance.net/experiment/tracking/detail?Id={baseline_project_id}&selectedTrial={baseline_run_id}&tab=CHART'  # noqa: E501
        result_lark_message += f'Current tracking: {current_tracking_link}\n'
        result_lark_message += f'Baseline tracking: {baseline_tracking_link}\n\n'

        # get all metrics
        conclusion = 0
        for processor in metric_processors:
            processor.process(baseline_project_name, run_id, baseline_run_id)
            conclusion = max(conclusion, processor.conclusion)
            result_lark_message += processor.get_analysis_str()

        msg_header = 'Results:\n'
        msg_header += f'Task Name: {meta_info["default_experiment_name"]}\n'
        msg_header += (f"Conclusion: {processor.conclusion_level[conclusion]}. "
                       f"{'' if conclusion == 0 else get_at_user_msg(owner_name)}\n")
        msg_header += f"{'*' * 150}\n"
        msg_header += "Details as below:\n"

        result_lark_message = msg_header + result_lark_message
        # Send another lark message of results
        send_message_to_alphaseed_ci_group(result_lark_message)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"GLOBAL_JOB_ID = {GLOBAL_JOB_ID}")
        print(f"GLOBAL_RH2_CLIENT = {GLOBAL_RH2_CLIENT}")
        if (GLOBAL_JOB_ID is not None and GLOBAL_RH2_CLIENT is not None and
                GLOBAL_TRIAL_STAUTS not in ['failed', 'finished', None]):
            # stop the trial
            print(f'create_merlin_trial exit abnormally! Stopping trial and Exception as below:\n{e}')
            stop_job(GLOBAL_RH2_CLIENT, GLOBAL_JOB_ID)
        raise e
