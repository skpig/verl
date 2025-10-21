from databus import collect_array
import json
import os

ARNOLD_TRIAL_ID = os.environ.get("ARNOLD_TRIAL_ID", "0")
ARNOLD_TRIAL_OWNER = os.environ.get("ARNOLD_TRIAL_OWNER", "0")
CHANNEL = "llm_rl_trace_hub"

if os.getenv("RUNTIME_IDC_NAME", "") == "wlby":
    CHANNEL = "llm_rl_trace_hub_wlby"


def send_to_kafka(message):
    message["ARNOLD_TRIAL_ID"] = ARNOLD_TRIAL_ID
    message["ARNOLD_TRIAL_OWNER"] = ARNOLD_TRIAL_OWNER
    collect_array(CHANNEL, [json.dumps(message, ensure_ascii=False).encode("utf-8")])
