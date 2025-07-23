import json
import torch
import inspect
from argparse import Namespace
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union, Iterable
from typing_extensions import Annotated, Literal, Required, TypedDict

from openai.types.completion_usage import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import (ChatCompletion, Choice, ChatCompletionMessage)
from openai.types.chat.chat_completion_message_tool_call import (ChatCompletionMessageToolCall, Function)

_MOCK_LONG_INFO = Namespace(min=-9223372036854775808, max=9223372036854775807)
_LONG_INFO = torch.iinfo(torch.long)
assert _LONG_INFO.min == _MOCK_LONG_INFO.min
assert _LONG_INFO.max == _MOCK_LONG_INFO.max


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None


class ResponseFormat(BaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = True


class ChatCompletionNamedFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ChatCompletionToolsParam(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ChatCompletionRolloutMessageParam(TypedDict, total=False):
    prompt: Required[Union[str, List[int]]]
    images_bytes_ref: Optional[str]
    image_data_ref: Optional[str]
    """The contents of the user message."""


class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: Union[ChatCompletionRolloutMessageParam, List[ChatCompletionMessageParam]]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"], Literal["auto"], ChatCompletionNamedToolChoiceParam]] = "none"

    # NOTE this will be ignored by VLLM -- the model determines the behavior
    parallel_tool_calls: Optional[bool] = False
    user: Optional[str] = None

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = 0
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    max_length: Optional[int] = None

    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    meta_info: Optional[Dict] = None

    # doc: end-chat-completion-extra-params

    def to_sampling_params(self):
        _AVAILABLE_SAMPLING_PARAMS = {
            "max_tokens": "max_new_tokens",
            "max_length": "max_length",
            "top_k": "top_k",
            "top_p": "top_p",
            "temperature": "temperature",
        }
        sampling_params = dict()
        for param_oai, param_xperf in _AVAILABLE_SAMPLING_PARAMS.items():
            if (value := getattr(self, param_oai, None)) is not None:
                sampling_params[param_xperf] = value
        return sampling_params


class ChatCompletionMessageRollout(ChatCompletionMessage):

    prompt: Optional[str] = None
    """input+output prompt"""
    raw_output_ids: Optional[List[int]] = None
    """The raw token ids of the message."""
    response_log_probs: Optional[List[float]] = None
    """The log probabilities of the message."""
    is_finished: bool = True
    """Whether the message is finished."""
    model_output_mask: Optional[List[bool]] = None
    """Mask indicating whether an output token is generated by the model decoding"""
    extra_data: Optional[dict] = None
    """Extra info required for trainer"""
    metrics: Optional[dict] = {}
    """Query level metrics"""
    image_data_ref: Optional[str] = None
    """Query image data"""
    num_image_tokens: Optional[List[int]] = None
    """Number of image tokens"""

    def to_dict(self):
        return self.model_dump()


class ChoiceRollout(BaseModel):

    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    index: int
    message: ChatCompletionMessageRollout  # 关键：期望我们的自定义消息类型
    logprobs: Optional[Any] = None

    def to_dict(self):
        return self.model_dump()


class ChatCompletionRollout(BaseModel):

    id: str
    choices: List[ChoiceRollout]  # 关键：使用我们的自定义Choice类型
    created: int
    model: str
    object: Literal["chat.completion"] = "chat.completion"
    system_fingerprint: Optional[str] = None
    usage: Optional[CompletionUsage] = None

    def to_dict(self):
        return self.model_dump()
