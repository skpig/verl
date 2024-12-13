import contextlib
import os
import time
import requests


def __update_config(trial_id, tensor_core_activity_first, dc):
    '''
    curl 'https://eurus-lq.byted.org/morningstar/api/v1/trials/35774201/config' \
    --header 'Content-Type: application/json' \
    --data '{"tensorCoreActivityFirst": True}'

    details of this API is shared:
    https://bytedance.larkoffice.com/docx/G9V4d1BD5oIzRjxL07ycmBx2nHg
    '''

    url = f'https://eurus-{dc}.byted.org/morningstar/api/v1/trials/{trial_id}/driver_config'
    headers = {'Content-Type': 'application/json'}
    data = {'tensorCoreActivityFirst': tensor_core_activity_first}

    retries = 5
    for attempt in range(retries):
        try:
            response = requests.patch(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f'update tensorcore collection config, attempt {attempt + 1} failed: {e}')
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retrying

    print(f'update tensorcore collection config all attempts failed')


def __turn_on_tensorcore_collect():
    if 'ARNOLD_ROBUST_TRAINING' in os.environ:
        trial_id = os.environ["ARNOLD_TRIAL_ID"]
        dc = os.environ["RUNTIME_IDC_NAME"]
        from datetime import datetime
        current_timestamp = datetime.now()
        formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f'turn on tensorcore collection {formatted_timestamp}')
        __update_config(trial_id, True, dc)


def __turn_off_tensorcore_collect():
    if 'ARNOLD_ROBUST_TRAINING' in os.environ:
        trial_id = os.environ["ARNOLD_TRIAL_ID"]
        dc = os.environ["RUNTIME_IDC_NAME"]
        from datetime import datetime
        current_timestamp = datetime.now()
        formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f'turn off tensorcore collection {formatted_timestamp}')
        __update_config(trial_id, False, dc)


@contextlib.contextmanager
def tensorcore_collection():
    __turn_on_tensorcore_collect()
    try:
        yield
    finally:
        __turn_off_tensorcore_collect()
