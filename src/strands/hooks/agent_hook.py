from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Protocol, TYPE_CHECKING, List, Type, Dict, Any, TypeVar, ParamSpec, Callable, Generic, \
    overload, Iterable

from strands.types.tools import AgentTool


if TYPE_CHECKING:
    from strands import Agent


class Reference(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

class AgentHook(Protocol):
    def register_hooks(self, hooks: "AgentHookManager", agent: "Agent") -> None: ...

T = TypeVar('T', bound=Callable)


@dataclass
class HookEvent(Protocol):
    agent: "Agent"

@dataclass
class AgentInitializedHookEvent(HookEvent):
    ...

@dataclass
class TextDiffsHookEvent(HookEvent):
    data: str

@dataclass
class ToolCallHookEvent(HookEvent):
    tool: AgentTool
    tool_input: str
    tool_response: str
    tool_response_metadata: dict[str, Any]

TEvent = TypeVar('TEvent', bound=HookEvent)


class HookCallback(Protocol, Generic(TEvent)):
    def __call__(self, event: TEvent) -> None: ...

OrderedHookCallback = namedtuple('OrderedHookCallback', ['callback', 'order'])

# Consider AgentContext (per agent), RequestContext (per turn), ToolContext (per tool)


class AgentHookManager:
    _registered_hooks: Dict[Type, List[OrderedHookCallback]]

    def __init__(self, agent: "Agent", parent: Optional["AgentHookManager"] = None, hooks: Optional[List[AgentHook]] = None) -> None:
        self._agent = agent
        self._parent = parent
        self._registered_hooks = {}

        for hook in hooks or []:
            self.add(hook)

    def add(self, hook: AgentHook):
        hook.register_hooks(hooks=self, agent=self._agent)

    def hook_into(self, hook_type: Type[T], callback: HookCallback[T], order: int = 0):
        if hook_type not in self._registered_hooks:
            self._registered_hooks[hook_type] = []

        self._registered_hooks[hook_type].append(OrderedHookCallback(callback=callback, order=order))

    def _add_callbacks_to_list(self, hook_id: Type[TEvent], callbacks: list[OrderedHookCallback]) -> list[
        OrderedHookCallback]:
        if self._parent is not None:
            self._parent._add_callbacks_to_list(hook_id, callbacks)

        my_callbacks = self._registered_hooks.get(hook_id)
        if my_callbacks is not None:
            callbacks.extend(my_callbacks)

        return callbacks

    def get_hook_callbacks(self, hook_id: Type[TEvent]) -> list[Callable[[TEvent], None]]:
        callbacks = self._add_callbacks_to_list(hook_id, [])
        sorted_callbacks = sorted(callbacks, key=lambda x: x.order)
        # Extract just the callback functions
        return [ordered_callback.callback for ordered_callback in sorted_callbacks]

    def invoke_hooks(self, event: TEvent) -> None:
        hook_type = type(event)

        for hook_callback in self.get_hook_callbacks(hook_type):
            hook_callback(event)


class TextPrinting(AgentHook):
    def register_hooks(self, hooks: "AgentHookManager", agent: "Agent") -> None:
        hooks.hook_into(TextDiffsHookEvent, self._handle_text_diffs)
        hooks.hook_into(ToolCallHookEvent, self._handle_tool_calls)

    def _handle_text_diffs(self, event: TextDiffsHookEvent) -> None:
        print(event.data)

    def _handle_tool_calls(self, event: ToolCallHookEvent) -> None:
        print(f"Tool call: {event.tool.tool_name} with input: {event.tool_input}")

# printer = TextPrinting()
# storage = DdbStorageProvider()
#
# >>> agent = Agent(hooks=[printer, storage])
