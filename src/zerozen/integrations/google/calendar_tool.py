# calendar_tool_v2.py - Calendar tool with dataclass credentials support
from __future__ import annotations
import datetime as dt
import typing as t

import backoff
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Import here to avoid circular imports
if t.TYPE_CHECKING:
    from .creds import CredentialRecord


CalendarEvent = dict[str, t.Any]


def _rfc3339(ts: str | dt.datetime | None) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        # Accept "YYYY-MM-DD" or full ISO8601; convert to RFC3339
        if len(ts) == 10:
            return ts + "T00:00:00Z"
        return ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.isoformat()


def _is_transient(err: Exception) -> bool:
    if isinstance(err, HttpError) and getattr(err, "resp", None):
        return err.resp.status in (429, 500, 502, 503, 504)
    return False


class CalendarTool:
    """
    Google Calendar integration with Google OAuth2 credentials.
    Accepts either Credentials or CredentialRecord objects.
    """

    def __init__(
        self, credentials: Credentials | CredentialRecord, *, logger: t.Any = None
    ):
        """
        Args:
            credentials: Google OAuth2 credentials with Calendar scope, or CredentialRecord
            logger: Optional logger instance
        """
        from .creds import CredentialRecord  # Import here to avoid circular imports

        if isinstance(credentials, CredentialRecord):
            self._credentials = credentials.to_credentials()
        else:
            self._credentials = credentials
        self._logger = logger
        self._service = None

    def _svc(self):
        """Get or create Calendar service."""
        if self._service is None:
            self._service = build(
                "calendar", "v3", credentials=self._credentials, cache_discovery=False
            )
        return self._service

    def list_events(
        self,
        *,
        calendar_id: str = "primary",
        time_min: str | dt.datetime | None = None,
        time_max: str | dt.datetime | None = None,
        max_results: int = 50,
        single_events: bool = True,
        order_by: str = "startTime",
        page_token: str | None = None,
    ) -> dict:
        """
        List Google Calendar events.

        Args:
            calendar_id: Calendar ID (default "primary")
            time_min: Lower bound (inclusive) for event start time
            time_max: Upper bound (exclusive) for event start time
            max_results: Max results (1-250)
            single_events: Whether to expand recurring events
            order_by: Sort order ("startTime" or "updated")
            page_token: For pagination

        Returns:
            {"events": [CalendarEvent, ...], "nextPageToken": str | None}
        """
        max_results = max(1, min(int(max_results or 50), 250))
        svc = self._svc()

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_time=20,
            jitter=None,
            giveup=lambda e: not _is_transient(e),
        )
        def _list():
            return (
                svc.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=_rfc3339(time_min),
                    timeMax=_rfc3339(time_max),
                    maxResults=max_results,
                    singleEvents=single_events,
                    orderBy=order_by,
                    pageToken=page_token,
                )
                .execute()
            )

        res = _list() or {}
        return {
            "events": res.get("items", []),
            "nextPageToken": res.get("nextPageToken"),
        }

    def get_event(
        self,
        *,
        event_id: str,
        calendar_id: str = "primary",
    ) -> dict:
        """
        Get a single Google Calendar event.

        Args:
            event_id: Event ID
            calendar_id: Calendar ID (default "primary")

        Returns:
            {"event": CalendarEvent}
        """
        svc = self._svc()

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_time=20,
            jitter=None,
            giveup=lambda e: not _is_transient(e),
        )
        def _get():
            return svc.events().get(calendarId=calendar_id, eventId=event_id).execute()

        return {"event": _get()}
