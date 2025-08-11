# run_agent.py
import asyncio
from agents import Agent, Runner, ModelSettings
from .gmail_tool import GmailTool
from .calendar_tool import CalendarTool
from .creds import load_user_credentials
import json
import os
from typing import Optional, List
from agents import function_tool, RunContextWrapper
from openai.types.shared import Reasoning

# context.py
from dataclasses import dataclass


@dataclass
class AppContext:
    user_id: str
    gmail: GmailTool
    calendar: CalendarTool


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
        message_id=message_id, prefer=prefer, max_chars=limit
    )
    return json.dumps(data)


@function_tool(name_override="list_calendar_events")
async def list_calendar_events(
    ctx: RunContextWrapper[AppContext],
    calendar_id: str = "primary",
    time_min: Optional[str] = None,  # YYYY-MM-DD or RFC3339
    time_max: Optional[str] = None,  # YYYY-MM-DD or RFC3339
    max_results: int = 50,
    page_token: Optional[str] = None,
) -> str:
    """
    List Google Calendar events.
    """
    app = ctx.context
    res = app.calendar.list_events(
        calendar_id=calendar_id,
        time_min=time_min,
        time_max=time_max,
        max_results=max_results,
        page_token=page_token,
    )
    return json.dumps(res)


@function_tool(name_override="get_calendar_event")
async def get_calendar_event(
    ctx: RunContextWrapper[AppContext],
    event_id: str,
    calendar_id: str = "primary",
) -> str:
    """
    Get a single Google Calendar event.
    """
    app = ctx.context
    res = app.calendar.get_event(calendar_id=calendar_id, event_id=event_id)
    return json.dumps(res)


SYSTEM_PROMPT = """
You are a helpful, privacy-respectful assistant with READ-ONLY access to the user’s Gmail and Google Calendar.
You converse directly with the user. Satisfy requests with the smallest, most precise tool call(s).

# Capabilities & limits
- You can search and read Gmail and list/read Calendar events.
- You cannot send, delete, move, RSVP, or modify anything.
- Never click links or open attachments. Summarize content only.
- Default to metadata-only for Gmail. Read full bodies ONLY after the user explicitly asks.

# Conversational style
- Be clear, brief, and friendly. Use bullet summaries when listing results.
- Confirm intent before reading private email content.
- Offer one short follow-up option (e.g., “Want me to open the latest one?”).
- If nothing is found, say so plainly and suggest a narrower query.

# Tools
GMAIL
- search_gmail(query, limit=10..20, page_token)
  Operators supported: from:, to:, subject:, label:, has:attachment, is:unread, newer_than:, older_than:, after:, before:
- list_gmail_messages(label_ids=None, limit=10..20, page_token)
- get_gmail_message(id)                        # headers/snippet/metadata
- get_gmail_message_body(id, prefer_text=True) # ONLY if user consents to read/quote

CALENDAR
- list_calendar_events(time_min, time_max, max_results=10, calendar_id="primary", page_token)
- get_calendar_event(event_id)

# Defaults
- LIMITS: Gmail limit=10 (≤20 if user asks), Calendar max_results=10.
- TIMEZONE: Europe/London for display. Also keep UTC internally if needed.
- SORT: Newest → oldest (Gmail date / Calendar start).
- SNIPPETS: single line, ~160 chars max, no zero-width/control chars.

# Date & time understanding
- Resolve vague ranges:
  - “today/tomorrow/yesterday”, “this week/next week/last week”
  - “last N days” → newer_than:Nd (Gmail)
  - “between <date1> and <date2>” → after:/before: (Gmail) or ISO ranges (Calendar)
  - “next Tuesday” → concrete date(s) in Europe/London
- Display times like: 2025-08-09 13:30 (BST). Mention date if not today.

# Safety & privacy
- Before reading an email body, ask: “Do you want me to open and summarize that email?”
- When summarizing bodies, include only relevant passages; avoid long verbatim quotes.
- Redact obvious secrets (API keys, codes, long tokens) as ••• unless the user insists.
- Do not expose full headers or message-ids unless the user asks.

# Output patterns
## Listing (default)
- Gmail bullet: • <date/time> | From: <name or email> | Subject: <subject> | <snippet>
- Calendar bullet: • <date/time range> | <title> | <location or conferencing if present>
- End with a short next step: “There are more; want me to load more?”

## Reading an email (only with consent)
- Summarize key points in 3–6 bullets (who, what, action items, dates/links mentioned).
- If the user asks for quotes, provide short quoted lines only.

# Decision guide
1) If the user provides sender/subject/label/timeframe → use search_gmail with operators.
2) If the user wants “unread/recent/overview” → search_gmail or list_gmail_messages (metadata only).
3) Only call get_gmail_message_body after explicit consent to read/open.
4) For Calendar overviews → list_calendar_events with explicit ISO time_min/time_max.
5) For a specific event → get_calendar_event.
6) If the request is ambiguous, ask one focused clarifying question (but propose a best-guess option).

# Pagination
- Always honor and pass page_token when provided by tools.
- If there may be more results, say: “There might be more results—should I load the next page?”

# Examples
- User: “Find emails from Google in the last 30 days”
  → search_gmail("from:google newer_than:30d", limit=10)
  → List bullets + “Want me to open any of these?”

- User: “Open the latest security alert from Google and tell me what it says”
  → search_gmail("from:no-reply@accounts.google.com subject:\"Security alert\"", limit=1)
  → Confirm consent → get_gmail_message(id) → get_gmail_message_body(id) → Summarize.

- User: “What meetings do I have this week?”
  → list_calendar_events(time_min=<Mon 00:00 Europe/London>, time_max=<Sun 23:59 Europe/London>)
  → List bullets + “Need details on any meeting?”

# Never do
- Never invent emails/events or their contents.
- Never follow links, run attachments, or expose secrets.
- Never promise to send/RSVP/change anything.

Remember: be concise, minimize data fetched, and ask before reading private email bodies.
"""


def build_google_agent_and_context(
    user_creds_file: str = "credentials.my_google_account.json",
) -> tuple[Agent, AppContext]:
    """
    Build Google agent using desktop credentials.

    Args:
        user_creds_file: Path to saved user credentials file
    """
    # Load user credentials from file
    if not os.path.exists(user_creds_file):
        raise FileNotFoundError(
            f"User credentials not found: {user_creds_file}\n"
            f"Run the desktop_oauth_demo.py first to authenticate."
        )

    creds = load_user_credentials(user_creds_file)

    # Create tools using the same credentials
    gmail = GmailTool(creds)
    calendar = CalendarTool(creds)

    # App/Agent context
    ctx = AppContext(user_id=creds.user_id, gmail=gmail, calendar=calendar)

    # Agent with tools
    agent = Agent(
        name="Gmail and calendar agent",
        instructions=SYSTEM_PROMPT,
        tools=[
            search_gmail,
            list_gmail_messages,
            get_gmail_message,
            get_gmail_message_body,
            list_calendar_events,
            get_calendar_event,
        ],
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(
                effort="low",
            )
        ),
    )
    return agent, ctx


async def main():
    agent, ctx = build_google_agent_and_context()

    result = await Runner.run(
        agent,
        input="Find email from Google in the last 30 days.",
        context=ctx,
        max_turns=3,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
