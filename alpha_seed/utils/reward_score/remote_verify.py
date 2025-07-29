import asyncio
import copy
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any

import aiohttp
import requests

judge_template_v1 = """You are a helpful assistant who evaluates the correctness and quality of models' outputs.
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.

Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
CORRECT
INCORRECT
Just return \"CORRECT\" or \"INCORRECT\", with no text around it.

Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


<Original Question Begin>:
{question}
<Original Question End>


<Gold Target Begin>:
{gold_answer}
<Gold Target End>


<Predicted Answer Begin>:
{answer}
<Predicted End>


Judging the correctness of candidates' answers:
"""

judge_template_v2 = """===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth answer. Your task is to determine if the ground truth answer is present in the LLM's response. Please analyze the provided data and make a decision.

===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers – look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer:"

===Input Data===
- Question: {question}
- Predicted Answer: {answer}
- Ground Truth Answer: {gold_answer}

===Output JSON Format===
You **MUST** provide your final evaluation in the following json format:
```json
{{
    "explanation": "how you made the decision?",
    "is_correct": true/false (boolean)
}}
```
Please proceed with the evaluation.
"""

judge_template_v2_1 = """===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth answer. Your task is to determine if the ground truth answer is present in the LLM's response. Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers – look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {answer}
- Ground Truth Answer: {gold_answer}
===Output JSON Format===
You **MUST** provide your final evaluation in the following json format:
```json
{{
    "explanation": "Please first state the ground truth answer and predicted answer here, and then determine whether the predicted answer is correct",
    "is_correct": true/false (boolean)
}}
```
Please proceed with the evaluation."""


@dataclass
class OpenAIConfig:
    api_key: str = ""
    model: str = "Qwen2.5-32B-Instruct"
    base_url: list = field(
        default_factory=lambda: [f"http://[2605:340:cd51:a00:6add:dcd5:8e3a:705e]:8010/v1/chat/completions"])
    max_batch_size: int = 50
    max_concurrency: int = 10
    timeout: int = 60


class AsyncOpenAIClient:

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrency)

    async def process_batch(self, messages_list: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(0, len(messages_list), self.config.max_batch_size):
                batch = messages_list[i:i + self.config.max_batch_size]
                tasks.extend([self._make_request(session, messages) for messages in batch])

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return self._process_results(results)

    async def _make_request(self, session: aiohttp.ClientSession, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        headers = {
            'Authorization': 'Bearer None',
            "Content-Type": "application/json",
        }

        data = {
            "model": self.config.model,
            "messages": messages,
            'max_tokens': 2048,
            'n': 1,
            'stop': None,
            'frequency_penalty': 0.0,
            'top_p': 0.75,
            'temperature': 1.0
        }
        async with self.semaphore:
            try:
                base_url_idx = random.randint(0, len(self.config.base_url) - 1)
                base_url = self.config.base_url[base_url_idx]
                async with session.post(base_url, headers=headers, json=data, timeout=self.config.timeout) as response:

                    if response.status == 429:
                        # 处理速率限制
                        print("Rate limit exceeded. Retrying in 5 seconds...")
                        retry_after = int(response.headers.get("Retry-After", "5"))
                        await asyncio.sleep(retry_after)
                        return await self._make_request(session, messages)

                    response_data = await response.json()
                    return {
                        "success": True,
                        "data": response_data,
                        # "messages": messages
                    }
            except Exception as e:
                return {
                    "success": False,
                    "err": str(e),
                    # "messages": messages
                }

    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        return processed_results


async def main():
    config = OpenAIConfig(api_key="your-api-key", max_concurrency=1024, max_batch_size=1024)
    prompt = '美国现任总统是谁？'
    judge_prompt = judge_template.format(question=prompt, gold_answer='拜登', answer='Joe Biden')
    # 准备测试数据
    messages_list = [[{"role": "user", "content": judge_prompt}] for i in range(1)]

    client = AsyncOpenAIClient(config)
    start_time = time.time()

    results = await client.process_batch(messages_list)
    # 处理结果
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)

    print(f"总请求数: {total_count}")
    print(f"成功请求数: {success_count}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    # print(results)
    # 打印错误信息
    for result in results:
        if not result["success"]:
            print(f"错误: {result['err']}")


class ThreadPoolOpenAIClient:

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_concurrency)

    def process_batch(self, messages_list: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        futures = []
        for i in range(0, len(messages_list), self.config.max_batch_size):
            batch = messages_list[i:i + self.config.max_batch_size]
            futures.extend([
                self.thread_pool.submit(ThreadPoolOpenAIClient._make_request, messages, self.config)
                for messages in batch
            ])

        results = [future.result() for future in futures]
        return self._process_results(results)

    @classmethod
    def _make_request(self, messages: List[Dict[str, str]], config) -> Dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {config.api_key}',
            "Content-Type": "application/json",
        }

        data = {
            "model": config.model,
            "messages": messages,
            'max_tokens': 2048,
            'n': 1,
            'stop': None,
            'frequency_penalty': 0.0,
            'top_p': 0.75,
            'temperature': 1.0
        }

        try:
            base_url_idx = random.randint(0, len(config.base_url) - 1)
            base_url = config.base_url[base_url_idx]

            response = requests.post(base_url, headers=headers, json=data, timeout=config.timeout)

            if response.status_code == 429:
                print("Rate limit exceeded. Retrying in 5 seconds...")
                retry_after = int(response.headers.get("Retry-After", "0.5"))
                time.sleep(retry_after)
                return ThreadPoolOpenAIClient._make_request(messages, config)

            return {
                "success": True,
                "data": response.json(),
            }
        except Exception as e:
            return {
                "success": False,
                "err": str(e),
            }

    def make_multiple_request(self, messages: List[Dict[str, str]], config: OpenAIConfig):
        models = config.model.split(",")
        urls = config.base_url
        if len(models) == 1:
            new_config = copy.deepcopy(config)
            new_config.model = models[0]
            return [self._make_request(messages, new_config)]
        else:
            assert len(models) == len(urls), "models and urls must have the same length"
            futures = []
            for model, url in zip(models, urls):
                sub_config = copy.deepcopy(config)
                sub_config.model = model
                sub_config.base_url = [url]
                futures.append(self.thread_pool.submit(ThreadPoolOpenAIClient._make_request, messages, sub_config))

            results = [future.result() for future in futures]
            return results

    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        return processed_results


async def make_request(config: OpenAIConfig,
                       session: aiohttp.ClientSession,
                       messages: List[Dict[str, str]],
                       max_retry: int = 3) -> Dict[str, Any]:
    headers = {
        'Authorization': f'Bearer {config.api_key}',
        "Content-Type": "application/json",
    }

    data = {
        "model": config.model,
        "messages": messages,
        'max_tokens': 2048,
        'n': 1,
        'stop': None,
        'frequency_penalty': 0.0,
        'top_p': 0.75,
        'temperature': 1.0
    }

    tries = 0
    while True:
        tries += 1
        try:
            base_url_idx = random.randint(0, len(config.base_url) - 1)
            base_url = config.base_url[base_url_idx]
            async with session.post(base_url, headers=headers, json=data, timeout=config.timeout) as response:

                if response.status == 429:
                    # 处理速率限制
                    print("Rate limit exceeded. Retrying in 5 seconds...")
                    retry_after = int(response.headers.get("Retry-After", "5"))
                    await asyncio.sleep(retry_after)
                    return await make_request(config, session, messages)

                response_data = await response.json()
                return {
                    "success": True,
                    "data": response_data,
                    # "messages": messages
                }
        except Exception as e:
            last_exception = e
            await asyncio.sleep(1)

        if tries >= max_retry:
            return {
                "success": False,
                "data": last_exception,
            }


async def make_multiple_requests(config: OpenAIConfig, max_retry: int, messages: List[Dict[str, str]]):
    _models = config.model.split(",")
    _urls = config.base_url
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _model, _url in zip(_models, _urls):
            sub_config = copy.deepcopy(config)
            sub_config.base_url = [_url]
            sub_config.model = _model
            tasks.append(make_request(sub_config, session, messages, max_retry))

        results = await asyncio.gather(*tasks)

    return results


def main_sync():
    config = OpenAIConfig(api_key="your-api-key", max_concurrency=1024, max_batch_size=1024)
    prompt = '美国现任总统是谁？'
    judge_prompt = judge_template.format(question=prompt, gold_answer='拜登', answer='Joe Biden')
    # 准备测试数据
    messages_list = [[{"role": "user", "content": judge_prompt}] for i in range(1)]

    client = ThreadPoolOpenAIClient(config)
    start_time = time.time()
    # results = ThreadPoolOpenAIClient._make_request(messages_list[0], config)

    results = client.process_batch(messages_list)
    import pdb
    pdb.set_trace()
    # 处理结果
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)

    print(f"总请求数: {total_count}")
    print(f"成功请求数: {success_count}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    # print(results)
    # 打印错误信息
    for result in results:
        if not result["success"]:
            print(f"错误: {result['err']}")


def main_async():
    config = OpenAIConfig(base_url=["https://ark-cn-beijing.bytedance.net/api/v3/chat/completions"],
                          model="ep-20250701152058-5lfj5",
                          api_key="0f13cd53-02f8-46e4-af8b-fe3fe19315f3")

    message = [{"role": "user", "content": "测试接口"}]

    async def async_env_demo():
        async with aiohttp.ClientSession() as session:
            res = await make_request(config, session, message)
            print(res)

            print(f"=================================")
            res = await make_multiple_requests(config, 1, message)
            print(res)

    try:
        asyncio.run(async_env_demo())
    except Exception as exc:
        print(f"Captured {exc}")


if __name__ == "__main__":
    # asyncio.run(main())
    main_async()
    # import requests
    # base_url: str = f"http://[2605:340:cd51:a00:d072:c0fc:8244:795b]:8010/v1/chat/completions"
    # headers = {
    #         'Authorization': 'Bearer None',
    #         "Content-Type": "application/json",
    #     }
    # prompt = '美国现任总统是谁？'
    # judge_prompt = judge_template.format(question=prompt, gold_answer='拜登', answer='Joe Biden')
    # # 准备测试数据
    # messages_list = [
    #     [{"role": "user", "content": judge_prompt}]
    #     for i in range(1)
    # ]
    # data = {
    #     "model": "Qwen2.5-32B-Instruct",
    #     "messages": messages_list,
    #     'max_tokens': 2048, 'n': 1, 'stop': None, 'frequency_penalty': 0.0, 'top_p': 0.75, 'temperature': 1.0
    # }
    # response = requests.post(url=base_url,
    #               json=data,
    #               headers=headers,
    #               verify=False,
    #               timeout=60)
    # print(response.json())
