import contextvars
import threading

# 给threaded agent用的
current_ctx = threading.local()

current_agent = contextvars.ContextVar('current_agent')
current_agent_tracker = contextvars.ContextVar('current_agent_tracker')


# 给threaded agent用的
def set_current_agent_tracker(tracker):
    current_ctx.current_agent_tracker = tracker


def get_current_agent_tracker():
    return current_ctx.current_agent_tracker


def set_current_agent(agent_name: str):
    current_ctx.current_agent = agent_name


def get_current_agent() -> str:
    return current_ctx.current_agent
