# run_agent.py
import asyncio
from agents import Agent, Runner
from .gmail_tool import GmailTool
from .creds import desktop_creds_provider_factory  # from earlier
import json
from typing import Optional, List
from agents import function_tool, RunContextWrapper

# context.py
from dataclasses import dataclass


@dataclass
class AppContext:
    user_id: str
    gmail: GmailTool


@function_tool(name_override="search_gmail")
async def search_gmail(
    ctx: RunContextWrapper[AppContext],
    query: str,
    label_ids: Optional[List[str]] = None,
    after: Optional[str] = None,  # YYYY-MM-DD or ISO8601
    before: Optional[str] = None,  # YYYY-MM-DD or ISO8601
    limit: int = 20,
) -> str:
    """
    Search Gmail using Gmail query syntax (e.g., `from:alice has:attachment newer_than:7d`).
    Returns a compact list of message summaries.

    Args:
        query: Gmail search string.
        label_ids: Optional Gmail label IDs to filter by.
        after: Lower bound date (inclusive).
        before: Upper bound date (exclusive).
        limit: Max results (1-50).
    """
    app = ctx.context
    res = app.gmail.search_messages(
        user_id=app.user_id,
        query=query,
        label_ids=label_ids,
        after=after,
        before=before,
        limit=limit,
    )
    # Tool outputs must be strings; agent will get this as the tool result.
    return json.dumps(res)


@function_tool(name_override="list_gmail_messages")
async def list_gmail_messages(
    ctx: RunContextWrapper[AppContext],
    label_ids: Optional[List[str]] = None,
    q: Optional[str] = None,
    limit: int = 20,
    page_token: Optional[str] = None,
    include_spam_trash: bool = False,
) -> str:
    """
    List message IDs using Gmail's list API (fast, minimal data).

    Args:
        label_ids: Filter by label IDs.
        q: Optional Gmail query string.
        limit: Max results (1-100).
        page_token: For pagination.
        include_spam_trash: Whether to include spam and trash.
    """
    app = ctx.context
    res = app.gmail.list_messages(
        user_id=app.user_id,
        label_ids=label_ids,
        q=q,
        limit=limit,
        page_token=page_token,
        include_spam_trash=include_spam_trash,
    )
    return json.dumps(res)


@function_tool(name_override="get_gmail_message")
async def get_gmail_message(
    ctx: RunContextWrapper[AppContext],
    message_id: str,
) -> str:
    """
    Fetch a single Gmail message (metadata only).

    Args:
        message_id: Gmail message ID.
    """
    app = ctx.context
    res = app.gmail.get_message(
        user_id=app.user_id,
        message_id=message_id,
    )
    return json.dumps(res)


@function_tool(name_override="get_gmail_message_body")
async def get_gmail_message_body(
    ctx: RunContextWrapper[AppContext],
    message_id: str,
    prefer: str = "text",
    limit: int = 50000,
) -> str:
    """
    Fetch a single Gmail message body for UI rendering.

    Args:
        message_id: Gmail message ID.
        prefer: "text" or "html". Defaults to "text".
        limit: Max characters to return for the selected body.
    """
    app = ctx.context
    data = app.gmail.get_message_body(
        user_id=app.user_id, message_id=message_id, prefer=prefer, max_chars=limit
    )
    return json.dumps(data)


SYSTEM_PROMPT = """
You are an email copilot focused on fast, precise retrieval from Gmail.

TOOL SELECTION
- Prefer `search_gmail` for semantic queries (e.g., "invoices from Stripe newer_than:30d", "from:google is:unread").
- Prefer `list_gmail_messages` when the user asks for raw IDs, pagination, or a quick count under specific labels (cheap + fast).
    - Use `get_gmail_message` only after you have a specific message_id, and only if the user needs headers/snippet details not returned by search/list. Always metadata-only.
    - Use `get_gmail_message_body` only when the user asks to read/open a message content for UI display. Prefer text unless user asks for HTML.

QUERY CRAFTING
- Build precise Gmail queries using operators: from:, to:, cc:, subject:, has:attachment, is:unread, newer_than:, older_than:, after:, before:, label:.
- If dates are vague ("last month", "past week"), prefer newer_than:/older_than: where possible; otherwise use after:/before: with ISO dates.

PAGINATION & LIMITS
- Keep `limit` small (10–20) unless the user asks for more.
- For large result sets or "show me more", call `list_gmail_messages` with `page_token` to paginate.

DATA MINIMIZATION
- Avoid fetching full bodies; metadata is usually enough to summarize (From, To, Subject, Date, snippet).
- Only call `get_gmail_message` when the user asks to open/read a specific message.

RESULT PRESENTATION
- Return a short, readable summary. For each message: Date, From, Subject, and a brief snippet.
- If no results, say so and suggest a refined query.

SAFETY & TONE
- Never fabricate results. If a query seems ambiguous, ask a short clarifying question and propose a tightened query.
- Keep answers concise. Prefer bullet points for lists.

EXAMPLES
- "Find email from Google in the last 30 days" → call `search_gmail(query="from:google newer_than:30d", limit=10)`.
- "List IDs in my Receipts label" → call `list_gmail_messages(label_ids=[<RECEIPTS_ID>], limit=20)` and include `page_token` if the user wants more.
    - "Open the third result" → use its `id` with `get_gmail_message(message_id=...)`.
    - "Show me the email content" → use `get_gmail_message_body(message_id=..., prefer="text")`.
"""


def build_gmail_agent_and_context() -> tuple[Agent, AppContext]:
    # 1) Build creds provider ONCE (desktop or DB-backed)
    creds_provider = desktop_creds_provider_factory(
        credentials_file="credentials.json",
        token_file="token.json",
    )

    # 2) Construct the integration ONCE, inject provider
    gmail = GmailTool(creds_provider)

    # 3) App/Agent context (not visible to the model)
    ctx = AppContext(user_id="local", gmail=gmail)

    # 4) Agent with tool(s)
    agent = Agent(
        name="Mail Agent",
        instructions=SYSTEM_PROMPT,
        tools=[
            search_gmail,
            list_gmail_messages,
            get_gmail_message,
        ],  # tool reads ctx.context.gmail
        model="gpt-5",
    )
    return agent, ctx


async def main():
    agent, ctx = build_gmail_agent_and_context()

    result = await Runner.run(
        agent,
        input="Find email from Google in the last 30 days.",
        context=ctx,
        max_turns=3,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
