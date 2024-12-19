import collections
import json
import logging
import os
import pprint
import time
from typing import Dict

import requests
from bytedrh2.http_client import _get_rh2_client, authenticate

REQUEST_TIMEOUT = 120
logging.basicConfig(level=logging.INFO)


class Retry:

    def __init__(self,
                 max_retry_times: int = 5,
                 retry_interval_seconds: int = 5,
                 skip_exception_after_max_retries: bool = False):
        self.max_retry_times = max_retry_times
        self.retry_interval_seconds = retry_interval_seconds
        self.skip_exception_after_max_retries = skip_exception_after_max_retries

    def __call__(self, func):

        def wrapper(*args, **kwargs):
            for i in range(self.max_retry_times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == self.max_retry_times - 1:
                        if self.skip_exception_after_max_retries:
                            logging.warning(f"{func.__name__} failed, args: {args}, kwargs: {kwargs}. "
                                            f"Skip after trying {self.max_retry_times} times.")
                            return

                        logging.warning(f"{func.__name__} failed, args: {args}, kwargs: {kwargs}")
                        raise e
                    logging.warning(f"{func.__name__} failed, retrying...")
                    time.sleep(self.retry_interval_seconds)

        return wrapper


def email_to_user_id(user_email):
    email_to_id_map: Dict[str, str] = {
        'liujuncai@bytedance.com': "ou_3333c0f1df8bbd9e748e7a9dc562336f",
        'yueyu@bytedance.com': 'ou_ac16e308784c743ac6736c6316e9ef52'
    }
    return email_to_id_map.get(user_email, '')


def version_to_image_url(version):
    version_to_image_url_map: Dict[str, str] = {
        'v72': "hub.byted.org/reckon/data.reckon.mlx.image_5940:71d942a7b58da1bf52ee434a235764d2",
        'v78': "hub.byted.org/reckon/data.reckon.mlx.image_5940:5518fb1cee770a1d32c4c509be32410c",
        'v85': "hub.byted.org/reckon/data.reckon.mlx.image_5940:5d8d8b56cf680f0c733e63b516035223"
    }
    return version_to_image_url_map.get(version, None)


def get_at_user_msg(user):
    if len(user) == 0:
        return ''
    else:
        user_email = user + '@bytedance.com'
        user_id = email_to_user_id(user_email)
        return f'<at user_id="{user_id}">{user_email}</at>'


def get_arnold_hosts(group_id=448, token=None, empty_only=True, exclude_maintenance=True, verbose=False):
    """
    group_id: Arnold group id
    token: user's token, please refer to https://bytedance.feishu.cn/docx/doxcn2qjS78VDr2QHkTN0jVLA5c
    empty_only: only grab empty hosts, default is True
    exclude_maintenance: exclude maintained hosts, default is True
    """
    url = f'https://arnold-api.byted.org/api/v3/groups/{group_id}/resources'
    result = arnold_http_request(url=url, method='GET')
    group_info = result.json()
    if verbose:
        pprint.pprint(group_info)
    assert isinstance(group_info, list)
    cluster_host_ips = collections.defaultdict(list)
    for cluster in group_info:
        assert 'cluster' in cluster, '[cluster] field is not found in the results'
        assert 'id' in cluster['cluster'], '[id] field is not found in the results'
        assert 'nodes' in cluster, '[nodes] field is not found in the results'
        cluster_host_ips[cluster['cluster']['id']] = []
        if len(cluster['nodes']) > 0:
            for node in cluster['nodes']:
                if exclude_maintenance and node['maintenance'] != 'False':
                    continue
                # sometimes daemon program occupies some resources
                if empty_only and (node['allocated_resources']['gpus'] > 0 or node['allocated_resources']['mem'] > 1 or
                                   node['allocated_resources']['cpus'] > 1 or node['allocated_resources']['ports'] > 0):
                    continue
                assert 'host' in node, '[host] field is not found in the node'
                cluster_host_ips[cluster['cluster']['id']].append(node['host'])
    return cluster_host_ips


def get_arnold_trial_logs(trial_id, target_workers='0', stream='stdout', file_prefix="", overwrite=True):
    instances = get_arnold_trial_instances(trial_id)
    worker_ids = target_workers.split(',') if isinstance(target_workers, str) else target_workers

    downloaded_logs = []
    for ins in instances:
        instance_id, role, host = ins["id"], ins["role"], ins["host"]
        worker_id = role.split("-")[-1]
        if len(worker_ids) > 0 and worker_id not in worker_ids:
            continue
        streams = ['stdout', 'stderr'] if stream in ["all", ""] else [stream]
        for s in streams:
            log_file = f"{trial_id}-{role}-{host}-{s}.log"
            if file_prefix != "":
                log_file = f"{file_prefix}-" + log_file
            # logging.warning(log_file)
            log_path = os.path.join('/tmp/merlin_experiment_logs/', log_file)
            ret = download_arnold_instance_log(instance_id, stream=s, file_name=log_path, overwrite=overwrite)
            if ret is None:
                logging.warning(f"Failed to download log: {log_path}")
            else:
                downloaded_logs.append(log_path)

    return downloaded_logs if len(downloaded_logs) > 0 else None


def download_arnold_instance_log(instance_id, file_name, stream='stdout', overwrite=True):
    if not overwrite and os.path.exists(file_name):
        logging.warning(f"{file_name} has already been downloaded")
        return file_name

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # get log url
    url = f"https://arnold.byted.org/api/v3/instances/{instance_id}/locallog/?stream={stream}"
    ret = arnold_http_request(url=url, method='GET')
    download_url = ret.json()['url']
    if download_url.strip() == "":
        logging.warning(" no download url")
        return None

    # download log
    result = arnold_http_request(url=download_url, method='GET')
    with open(file_name, 'wb') as f:
        f.write(result.content)
        # f.close()
    return file_name


def get_arnold_trial_instances(trial_id):
    # url = f"https://arnold-api.byted.org/api/v3/trials/{trial_id}"
    url = "https://arnold-api.byted.org/api/v3/instances/"
    params = {"trial_id": trial_id, "page_size": 1000}
    result = arnold_http_request(url=url, method='GET', params=params)
    result_json = result.json()
    assert "results" in result_json
    instances = result_json['results']
    for i in instances:
        i["role"] = i["natural_id"].split("__")[-1].replace("_", "-")

    # when # of instances > page_size
    while 'next_link' in result_json and result_json['next_link'] is not None:
        print(result_json['next_link'])
        new_result = arnold_http_request(url=result_json['next_link'], method='GET', params=params)
        result_json = new_result.json()
        assert "results" in result_json
        new_instances = result_json['results']
        for i in new_instances:
            i["role"] = i["natural_id"].split("__")[-1].replace("_", "-")
            instances.append(i)

    return instances


def get_arnold_trial_info(trial_id):
    url = f"https://arnold-api.byted.org/api/v3/trials/{trial_id}"
    trial_info = arnold_http_request(url=url, method='GET')
    return trial_info.json()


@Retry(max_retry_times=1, retry_interval_seconds=300)
def arnold_http_request(url: str, method='GET', token=None, data=None, params=None):
    key = "MERLIN_EXPERIMENT_ARNOLD_API_TOKEN"
    # for compatible with binary search
    if 'CI_ACTOR' in os.environ:
        username = os.environ["CI_ACTOR"].upper().replace(".", "_")
        key += f'_{username}'

    myToken = os.getenv(key) if token is None else token
    head = {'Authorization': f'token {myToken}'}
    if method not in ['GET', 'get', 'POST', 'post']:
        logging.warning(f"Invalid request method {method}")
        return None
    result = (requests.get(url=url, headers=head, timeout=REQUEST_TIMEOUT, params=params)
              if method in ['GET', 'get'] else requests.post(url=url, headers=head, json=data, timeout=REQUEST_TIMEOUT))
    if not result.ok:
        raise Exception(f'Bad response: {method} {url}. result.text = {result.text}')
    return result


# ################################ rh2 apis #################################


def rh2_authenticate(host=None, user_name=None, token=None):
    """
    for user_name and token, please refer to
        https://bytedance.feishu.cn/wiki/wikcnQLj9f0EtkidUxHe5XwXw1g
    """
    host = 'rh2.bytedance.net' if host is None else host
    user_name = os.getenv("MERLIN_EXPERIMENT_USER_NAME") if user_name is None else user_name
    token = os.getenv("MERLIN_EXPERIMENT_TOKEN") if token is None else token
    authenticate(host=host, user_name=user_name, access_token=token)


def create_rh2_client():
    return _get_rh2_client()


def rh2_http_request(client, method: str, url: str, data=None, experiment_name=""):
    if method not in ['GET', 'get', 'POST', 'post']:
        logging.warning(f"Experiment({experiment_name}): invalid request method {method}")
        return None
    if method in ['POST', 'post'] and (data is None or not (isinstance(data, dict) or isinstance(data, str))):
        logging.warning(f"Experiment({experiment_name}): invalid POST data: {data}")
        return None
    if data is not None and isinstance(data, dict):
        data = json.dumps(data)

    result = (client.http_call(method='GET', path=url)
              if method in ['GET', 'get'] else client.http_call(method='POST', path=url, data=data))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 200, f'Experiment({experiment_name}) {method} {url} failed (return code {result[0]})'
    assert isinstance(result[1], str)
    return result[1]


def get_trial_id_from_job_info(job_info):
    trial_id = job_info['jobRun']['meta']['arnoldJobInfo']['trialId']
    return trial_id


def get_run_id_from_exp_name(project_id, experiment_name):
    from wandb.sdk.internal.tracking_client import Client

    client = Client()
    result = client.list_runs(project_id=project_id, name=experiment_name)
    runs = []
    for run in result.get("List"):
        if run["Name"] != experiment_name:
            continue
        runs.append(run)
    if len(runs) > 1:
        raise Exception(f'run_name: "{experiment_name}" is ambiguous, at least {len(runs)} runs have this name')
    if len(runs) == 0:
        logging.warning(
            f'run_name: "{experiment_name}" can\'t be found in project_id: "{project_id}". Will return None. '
            f'It is possible that the tracking has not been created by the trial yet, '
            f'or the trial created tracking in another project.')
        return None
    logging.info(f'run_name: "{experiment_name}" found. Will return {runs[0]["RunId"]}')
    return runs[0]['RunId']


def get_run_id_from_job_info(job_info, target_workers='0'):
    # run_id is
    trial_id = job_info['jobRun']['meta']['arnoldJobInfo']['trialId']
    print(f'Trial id {trial_id}')
    # get rank 0 stderr
    key = 'View run at https://ml.bytedance.net/experiment/tracking/detail?Id='

    if '-' in target_workers:
        start, end = target_workers.split('-')
        target_workers = [i for i in range(int(start), int(end) + 1)]

    print(target_workers)

    for target_worker in target_workers:
        target_worker = str(target_worker)
        log = get_arnold_trial_logs(trial_id=trial_id, target_workers=target_worker, stream='stderr')
        assert len(log) == 1

        with open(log[0]) as f:
            output = f.read()

        if key in output:
            print(f'key in worker {target_worker}')
            break

    with open(log[0]) as f:
        output = f.read()

    lines = output.split('\n')
    key = 'View run at https://ml.bytedance.net/experiment/tracking/detail?Id='
    split = '&selectedTrial='
    for line in lines:
        # This is hack
        if key in line:
            print(line)
            break

    project_id_split_run_id = line.split(key)[-1]
    project_id, run_id = project_id_split_run_id.split(split)
    return project_id, run_id


def query_deprecated(project_id, run_id, name):
    from bytedds import DatasetOpenApiQueryClient

    client = DatasetOpenApiQueryClient(api_path='/7213628157906158652')

    # sql = f"""
    # select * from tracking_run_entity where run_id = '{run_id}' and name = 'training/loss'
    # """.strip()

    sql = f"""
    select * from tracking_run_entity where run_id in ('{run_id}', '{project_id}/{run_id}') and name = '{name}'
    """.strip()

    result = client.execute(sql)
    ret = {}
    count = {}

    for r in result:
        step = r['step']
        item = json.loads(r['item'])
        scalar = item['scalar']['value']
        if step not in ret:
            ret[step] = 0
            count[step] = 0
        ret[step] += scalar
        count[step] += 1.0

    for k in ret.keys():
        ret[k] /= count[k]

    return ret


@Retry(max_retry_times=5, retry_interval_seconds=300)
def query(project_id, run_id, name):
    import wandb

    api = wandb.TrackingApi()
    run = api.run(project=project_id, run_id=run_id)
    h = run.history(name=[name])
    step = h['step']
    data = h[name]
    ret = {}
    for s, d in zip(step, data):
        ret[s] = d
    return ret


def query_tracking_running_steps(project_id, run_id, name):
    import wandb

    api = wandb.TrackingApi()
    try:
        run = api.run(project=project_id, run_id=run_id)
        h = run.history(name=[name])
        step = h['step']
        min_step = min(step)
        max_step = max(step)
        return max_step - min_step
    except Exception as e:
        logging.warning(f"Failed to get steps from tracking api, {e}. That means: the "
                        f"merlin trial has not uploaded '{name}' to merlin tracking yet. "
                        f"It's worthy noting that it might not be exception. \n"
                        f"You can do some checks as below:\n"
                        f"  1. Reduce the value of `--trainer.log_every_n_steps`.\n"
                        f"  2. Check logs of the {run_id} to make sure the trial is not hang abnormally.")
        return 0


def launch_job(rh2_client, task_config):
    ret_str = rh2_http_request(rh2_client, method='POST', url='api/v1/job_run/launch', data=json.dumps(task_config))
    ret_info = json.loads(ret_str)
    assert 'jobRunId' in ret_info
    job_id = ret_info['jobRunId']
    return job_id


@Retry(max_retry_times=5, retry_interval_seconds=60, skip_exception_after_max_retries=True)
def stop_job(rh2_client, job_id):
    data = {'job_run_id': job_id, 'stop_by': os.getenv('MERLIN_EXPERIMENT_USER_NAME'), 'err_msg': ""}
    ret_str = rh2_http_request(rh2_client, method='POST', url=f'api/v1/job_run/stop/{job_id}', data=data)
    return ret_str


def get_job_run_info(rh2_client, job_id, experiment_name=''):
    if job_id is None:
        logging.warning(f"Experiment ({experiment_name}) has not been successfully launched yet.")
        return None
    ret_str = rh2_http_request(rh2_client, method='GET', url=f'api/v1/job_run/get/{job_id}')
    job_info = json.loads(ret_str)
    return job_info


def fork_job(job_info):
    """
    Fork a job from an existing job and return a job template.
    """
    new_job = {}
    job_run = job_info['jobRun']
    new_job['mlxlabRepoId'] = job_run['mlxlabRepoId']
    new_job['namespace'] = job_run['namespace']
    new_job['jobDefVersion'] = job_run['meta']['jobDefVersion']
    new_job['jobDefVersionNum'] = job_run['meta']['jobDefVersionNum']
    new_job['jobRunParams'] = job_run['meta']['jobRunParams']
    new_job['caption'] = job_run['meta']['caption']  # copy caption
    return new_job


if __name__ == '__main__':
    # export MERLIN_EXPERIMENT_TOKEN=xxx
    # export MERLIN_EXPERIMENT_USER_NAME=xxx
    rh2_authenticate()
    job_id = 'a74f97471ca49e37'
    rh2_client = create_rh2_client()
    job_info = get_job_run_info(rh2_client, job_id)
    print(job_info['jobRun']['meta']['jobDefVersion']['entrypointFullScript'])
