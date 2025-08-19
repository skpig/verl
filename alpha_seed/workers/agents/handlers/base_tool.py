from dataclasses import dataclass
from typing import Any

from alpha_seed.workers.agents.monitor_ctx import current_agent_tracker, current_agent
from alpha_seed.workers.agents.monitoring import AgentTaskTracker
from verl.tools.base_tool import BaseTool as OSSBaseTool


@dataclass
class ToolResult:
    result: Any  # 工具的结果返回值
    retries: int = 0  # 调用过程中重试的次数，如果这个值等于max_attempts-1表示达到最大重试次数了，如果是0表示1次成功
    max_attempts: int = 0  # 最尝试次数，工具超过这个尝试次数应该放弃尝试
    success: bool = True  # 无论重试多少次，只要最后结果给到下一轮llm的就算成功，除非明确知道这个调用失败，否则默认按成功处理
    # 其他字段按需添加


class BaseTool(OSSBaseTool):

    # 注意这个函数的返回值类型变了
    # 注意前两个参数instance_id和parameters不能改，其他自定义的参数加到kwargs里
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> ToolResult:
        raise NotImplementedError()

    def __getattribute__(self, name):
        if name == "execute":
            orig_call = super().__getattribute__(name)

            async def wrapped_execute(*args, **kwargs):
                try:
                    tracker: AgentTaskTracker = current_agent_tracker.get()
                except LookupError:
                    return await orig_call(*args, **kwargs)

                with tracker.tool_call(self.__class__.__name__):
                    result: ToolResult = await orig_call(*args, **kwargs)
                    if isinstance(result, ToolResult):
                        exceeded_max_attempts = result.retries == result.max_attempts - 1
                        tracker.incr_tool_call_counter(
                            self.__class__.__name__,
                            result.retries,
                            exceeded_max_attempts,
                            result.success,
                        )
                    return result

            return wrapped_execute
        return super().__getattribute__(name)
