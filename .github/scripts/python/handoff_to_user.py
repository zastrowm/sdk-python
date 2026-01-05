from rich.markup import escape
from rich.panel import Panel
from strands import tool
from strands.types.tools import ToolContext
from strands_tools.utils import console_util

from github_tools import add_issue_comment


@tool(context=True)
def handoff_to_user(
    message: str,
    tool_context: ToolContext,
    post_comment: bool,
    issue_number: int | None = None,
) -> str:
    """
    Hand off control to the user with a message. This stops the agent execution
    and waits for the user to respond before continuing.

    Args:
        message: The message to give to the user
        post_comment: If true, post the message as a comment on the GitHub issue/PR.
            Only set this to true when user intervention or feedback is required
            before continuing (e.g., clarification needed, approval required,
            or a decision must be made). Do not post a comment for simple status updates
            or completion messages. If you are asking a question to the user this MUST
            be true.
        issue_number: The issue or PR number to comment on (required if post_comment is true)

    Returns:
        The users response after handing back control
    """
    console = console_util.create()
    
    console.print(
        Panel(
            escape(message),
            title="[bold yellow]🤝 Handoff to User",
            border_style="yellow",
        )
    )
    
    # Post comment to GitHub if requested
    if post_comment:
        if issue_number is None:
            console.print(
                Panel(
                    "Cannot post comment: issue_number is required when post_comment is true",
                    title="[bold red]Error",
                    border_style="red",
                )
            )
        else:
            add_issue_comment(issue_number, message)
    
    request_state = {
        "stop_event_loop": True
    }
    tool_context.invocation_state["request_state"] = request_state

    # Return an empty string as this will break out of the event loop
    return ""