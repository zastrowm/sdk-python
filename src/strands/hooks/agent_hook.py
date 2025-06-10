from typing import Optional, Protocol, TYPE_CHECKING, List, Type, Dict, Any, TypeVar, ParamSpec, Callable, Generic

from strands.types.tools import AgentTool


if TYPE_CHECKING:
    from strands import Agent

T = TypeVar('T')

class Reference(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

class AgentInitialized(Protocol):
    def __call__(self, *, agent: "Agent") -> None: ...

class ToolTransformer(Protocol):
    def __call__(self, agent: "Agent", tool: Reference[AgentTool]) -> None: ...

class AgentHook(Protocol):
    def register_hooks(self, hooks: "AgentHookManager", agent: "Agent") -> None: ...


class AgentHookManager:

    registered_hooks: Dict[Type, List[Any]] = {}

    def __init__(self, agent: "Agent",  hooks: Optional[List[AgentHook]] = None) -> None:
        self.agent = agent
        self.hooks = hooks

    def add(self, hook: AgentHook):
        hook.register_hooks(hooks=self, agent=self.agent)

    def add_hook(self, hook_type: T, callback: T):
        if hook_type not in self.registered_hooks:
            self.registered_hooks[hook_type] = []

        self.registered_hooks[hook_type].append(callback)

    def get_hook(self, hook_type: Type[T]) -> T:
        return lambda *args, **kwargs: self._invoke_hook(hook_type, *args, **kwargs)

    def _invoke_hook(self, hook_type: Type[Callable], *args: Any, **kwargs: Any) -> None:
        if hook_type not in self.registered_hooks:
            return

        for hook in self.registered_hooks[hook_type]:
            hook(*args, **kwargs)

        return




