import traceback
from dataclasses import dataclass
from typing import Any

from alpha_seed.workers.agents.monitor_ctx import current_agent_tracker, current_agent
from verl.tools.base_tool import BaseTool as OSSBaseTool


@dataclass
class ToolResult:
    result: Any  # 工具的结果返回值
    retries: int = 0  # 调用过程中重试的次数，如果这个值等于max_attempts-1表示达到最大重试次数了，如果是0表示1次成功
    max_attempts: int = 0  # 最尝试次数，工具超过这个尝试次数应该放弃尝试
    success: bool = True  # 无论重试多少次，只要最后结果给到下一轮llm的就算成功，除非明确知道这个调用失败，否则默认按成功处理
    error_msg: str = ""  # 当tool call发生error时记录其str(e)
    error_traceback: str = ""  # 当tool call发生error时记录error的stack

    # 其他字段按需添加

    def to_serializable(self) -> 'ToolResult':
        if isinstance(self.result, str):
            safe_result = self.result
        else:
            try:
                safe_result = str(self.result)
            except Exception as e:
                tb = traceback.format_exc()
                safe_result = f"exception during converting ToolResult.result to string: {e}"
                self.error_msg += "\n" + safe_result
                self.error_traceback += "\n---\n" + tb
        return ToolResult(safe_result, self.retries, self.max_attempts, self.success, self.error_msg,
                          self.error_traceback)


class BaseTool(OSSBaseTool):

    # 注意这个函数的返回值类型变了
    # 注意前两个参数instance_id和parameters不能改，其他自定义的参数加到kwargs里
    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> ToolResult:
        raise NotImplementedError()

    def __getattribute__(self, name):
        if name == "execute":
            orig_call = super().__getattribute__(name)

            async def wrapped_execute(instance_id, parameter, **kwargs):
                from alpha_seed.workers.agents.monitoring import AgentTaskTracker
                try:
                    tracker: AgentTaskTracker = current_agent_tracker.get()
                except LookupError:
                    return await orig_call(instance_id, parameter, **kwargs)

                with tracker.tool_call(self.__class__.__name__, instance_id, parameter) as capturer:
                    try:
                        result: ToolResult = await orig_call(instance_id, parameter, **kwargs)
                        if isinstance(result, ToolResult):
                            capturer.capture_output(result)
                    except Exception as e:
                        tb = traceback.format_exc()
                        capturer.capture_exception(e, tb)
                        raise
                    return result

            return wrapped_execute
        return super().__getattribute__(name)
