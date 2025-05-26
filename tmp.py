# launch the offline engine
import asyncio
import io
import os

from PIL import Image
import requests
import sglang as sgl
import asyncio

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge
from sglang.srt.sampling.sampling_params import SamplingParams


if __name__ == "__main__":
    if is_in_ci():
        import patch
    else:
        import nest_asyncio

        nest_asyncio.apply()
        print("nest_asyncio applied")


    engine = sgl.Engine(model_path='/pretrain/Qwen/Qwen2.5-0.5B', skip_tokenizer_init=True)
    sampling_params = {'n': 2}
    a = asyncio.run(engine.async_generate(input_ids=[[111,222,333], [444,555]], sampling_params=sampling_params, return_logprob=True))
    print(a)