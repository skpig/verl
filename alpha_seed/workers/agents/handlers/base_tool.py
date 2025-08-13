from alpha_seed.workers.agents.monitor_ctx import current_agent_tracker, current_agent
from alpha_seed.workers.agents.monitoring import AgentTaskTracker
from verl.tools.base_tool import BaseTool as OSSBaseTool


class BaseTool(OSSBaseTool):

    def __getattribute__(self, name):
        if name == "execute":
            orig_call = super().__getattribute__(name)

            async def wrapped_execute(*args, **kwargs):
                try:
                    tracker: AgentTaskTracker = current_agent_tracker.get()
                    agent_class_name = current_agent.get()
                except LookupError:
                    return await orig_call(*args, **kwargs)

                with tracker.tool_call(agent_class_name, self.__class__.__name__):
                    result = await orig_call(*args, **kwargs)
                    return result

            return wrapped_execute
        return super().__getattribute__(name)
