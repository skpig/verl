import ray
import asyncio
import logging
import random

from bytedagi.model_io import (
    InferenceRequest,
    ModelIO,
    ChatUltramanLLMServer,
)
from langchain.schema import HumanMessage, SystemMessage


@ray.remote
class BaseLLMRemoteClient:

    def __init__(
        self,
        psm,
        idc,
        cluster,
        model_name,
        max_response_length=1024,
        pool_size=4,
    ):
        self.clients = [
            self._create_client(psm, idc, cluster, model_name, max_response_length) for _ in range(pool_size)
        ]

    def _create_client(
        self,
        psm,
        idc,
        cluster,
        model_name,
        max_response_length,
        top_p=0.7,
        lb_type="WeightedConsistentHash",
    ):
        llm_server = ChatUltramanLLMServer(
            psm=psm,
            idc=idc,
            cluster=cluster,
            model_name=model_name,
            max_new_tokens=max_response_length,
            top_p=top_p,
            lb_type=lb_type,
        )
        return ModelIO(llm_server)

    async def generate(self, prompt, system_prompt, max_retry=2, retry_interval=1):
        for attempt in range(max_retry):
            try:
                client = random.choice(self.clients)
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))
                response = await client.astream(InferenceRequest(messages=messages))
                response_buffer = []
                async for msg in response.astream_output:
                    response_buffer.append(msg.content)
                response = "".join(response_buffer)
                return {"response": response, "status": "success"}
            except Exception as e:
                logging.warning(f"[GRM Request Attempt] {attempt+1} failed: {e}")
                await asyncio.sleep(retry_interval)

        return {"response": "", "status": "fail"}
