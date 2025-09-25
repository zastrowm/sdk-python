"""Experimental hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.
"""

from typing import TypeAlias

from ...hooks.events import AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent, BeforeToolCallEvent

BeforeToolInvocationEvent: TypeAlias = BeforeToolCallEvent
AfterToolInvocationEvent: TypeAlias = AfterToolCallEvent
BeforeModelInvocationEvent: TypeAlias = BeforeModelCallEvent
AfterModelInvocationEvent: TypeAlias = AfterModelCallEvent
