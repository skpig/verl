import ray
import json
import requests
import time
from sandbox_fusion import submit, SubmitRequest, TestConfig
from bytedance import servicediscovery
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

import requests
from tenacity import retry, stop_after_attempt, wait_incrementing

OJ_MAX_ATTEMPTS = 3
CLIENT_TIMEOUT = 120


def get_sandbox_endpoint(aider_service_psm):
    sd_result = servicediscovery.get_one(aider_service_psm, address_family="dual-stack")
    host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
    port = sd_result["Port"]
    endpoint = f"http://{host}:{port}"
    return endpoint


def compute_score_client(solution_str, ground_truth, aider_service_psm, data_uid, config, **argv) -> float:
    """Directly retrieve the scores from SandboxClient"""
    score = None
    if config.trainer.use_remote_sandbox:
        # get the sandbox client endpoint
        handler = ray.get_actor('remote_client')
        # retrieve the score directly
        score = ray.get(handler.get_results.remote(data_uid))

    if score is None:
        score = compute_score(solution_str, ground_truth, aider_service_psm, **argv)

    # optionally, compute the score with original code to compare the results
    # score_original = compute_score(solution_str, ground_truth, code_sandbox_psm, **argv)
    # assert score == score_original

    return score


class AiderV2Result(BaseModel):
    accepted: bool
    extracted_code: str
    stdout: str
    stderr: str


def before_retry_sleep(s):
    print(f'error requesting faas for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}')


def on_retry_error(s):
    e = s.outcome.exception()
    raise e


def evaluate(url, item: dict) -> AiderV2Result:
    response = requests.post(f'{url}/evaluate', json=item, timeout=CLIENT_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f'[Aider V2] responded with code {response.status_code}: {response.text}. request: {item}')
    return AiderV2Result(**response.json())


def compute_score(solution_str, ground_truth, aider_service_psm, **argv) -> float:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    for run in range(OJ_MAX_ATTEMPTS):
        endpoint = get_sandbox_endpoint(aider_service_psm)
        req = {
            "language": ground_truth["language"],
            "name": ground_truth["name"],
            "completion": solution_str,
            "is_training": ground_truth.get("is_training", False)
        }
        try:
            result = evaluate(endpoint, req)
            if result.accepted:
                return 1
            else:
                return -1
        except Exception as ex:
            print(f'sandbox fail with error: {ex}, retrying with {run+1}/{OJ_MAX_ATTEMPTS} attempts')
    print(f'Finally aider sandbox fails')
    return -2


def test_compute_score():
    print(
        compute_score(solution_str='''```rust
use std::vec::Vec;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    IncompleteNumber,
}

/// Convert a list of numbers to a stream of bytes encoded with variable length encoding.
pub fn to_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &num in values {
        let mut value = num;
        let mut chunks = Vec::new();
        loop {
            let chunk = value & 0x7F;
            chunks.push(chunk);
            value >>= 7;
            if value == 0 {
                break;
            }
        }
        chunks.reverse();
        let len = chunks.len();
        for i in 0..len {
            let mut byte = chunks[i] as u8;
            if i != len - 1 {
                byte |= 0x80;
            }
            bytes.push(byte);
        }
    }
    bytes
}

/// Given a stream of bytes, extract all numbers which are encoded in there.
pub fn from_bytes(bytes: &[u8]) -> Result<Vec<u32>, Error> {
    let mut result = Vec::new();
    let mut current_value = 0u32;
    let mut current_bits = 0;
    for &byte in bytes {
        let chunk = byte & 0x7F;
        current_value = (current_value << 7) | chunk as u32;
        current_bits += 7;
        if (byte & 0x80) == 0 {
            result.push(current_value);
            current_value = 0;
            current_bits = 0;
        }
    }
    if current_bits != 0 {
        Err(Error::IncompleteNumber)
    } else {
        Ok(result)
    }
}
```''',
                      ground_truth={
                          "language": "rust",
                          "name": "variable_length_quantity"
                      },
                      aider_service_psm="data.aml.code_jiaze_test.service.hl"))


if __name__ == '__main__':
    # pass
    test_compute_score()
