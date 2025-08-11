# gmail_tool_v2.py - Gmail tool with dataclass credentials support
from __future__ import annotations
import datetime as dt
import typing as t

import backoff
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64

# Import here to avoid circular imports
if t.TYPE_CHECKING:
    from .creds import CredentialRecord


MessageSummary = dict[str, t.Any]


def _iso_to_epoch(iso: str) -> int:
    """Accepts 'YYYY-MM-DD' or any ISO8601; treats naive as UTC."""
    if not iso:
        return 0
    if len(iso) == 10:  # 'YYYY-MM-DD'
        iso = iso + "T00:00:00Z"
    clean = iso[:-1] if iso.endswith("Z") else iso
    d = dt.datetime.fromisoformat(clean)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def _metadata_headers() -> list[str]:
    return ["From", "To", "Cc", "Subject", "Date"]


def _normalize_metadata(msg: dict) -> MessageSummary:
    payload = msg.get("payload") or {}
    headers = {
        h["name"].lower(): h.get("value", "") for h in payload.get("headers", [])
    }
    return {
        "id": msg["id"],
        "threadId": msg.get("threadId"),
        "subject": headers.get("subject", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "date": headers.get("date", ""),
        "snippet": msg.get("snippet", ""),
        "labels": msg.get("labelIds", []),
    }


def _build_query(query: str, after: str | None, before: str | None) -> str:
    parts: list[str] = []
    if query:
        parts.append(query)
    if after:
        parts.append(f"after:{_iso_to_epoch(after)}")
    if before:
        parts.append(f"before:{_iso_to_epoch(before)}")
    return " ".join(parts).strip()


def _is_transient(err: Exception) -> bool:
    if isinstance(err, HttpError) and getattr(err, "resp", None):
        return err.resp.status in (429, 500, 502, 503, 504)
    return False


class GmailTool:
    """
    Gmail integration with Google OAuth2 credentials.
    Accepts either Credentials or CredentialRecord objects.
    """

    def __init__(
        self, credentials: Credentials | CredentialRecord, *, logger: t.Any = None
    ):
        """
        Args:
            credentials: Google OAuth2 credentials with Gmail scope, or CredentialRecord
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
        """Get or create Gmail service."""
        if self._service is None:
            self._service = build(
                "gmail", "v1", credentials=self._credentials, cache_discovery=False
            )
        return self._service

    def search_messages(
        self,
        *,
        query: str,
        label_ids: list[str] | None = None,
        after: str | None = None,  # 'YYYY-MM-DD' or ISO8601
        before: str | None = None,  # 'YYYY-MM-DD' or ISO8601
        limit: int = 20,
    ) -> dict:
        """
        Search Gmail messages with metadata.

        Args:
            query: Gmail search string
            label_ids: Optional Gmail label IDs to filter by
            after: Lower bound date (inclusive)
            before: Upper bound date (exclusive)
            limit: Max results (1-50)

        Returns:
            {"messages": [MessageSummary, ...]}
        """
        limit = max(1, min(int(limit or 20), 50))
        q = _build_query(query, after, before)
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
                svc.users()
                .messages()
                .list(
                    userId="me",
                    q=q,
                    labelIds=label_ids or [],
                    maxResults=limit,
                )
                .execute()
            )

        res = _list()
        ids = [m["id"] for m in res.get("messages", [])]

        summaries: list[MessageSummary] = []

        for mid in ids:

            @backoff.on_exception(
                backoff.expo,
                Exception,
                max_time=20,
                jitter=None,
                giveup=lambda e: not _is_transient(e),
            )
            def _get():
                return (
                    svc.users()
                    .messages()
                    .get(
                        userId="me",
                        id=mid,
                        format="metadata",
                        metadataHeaders=_metadata_headers(),
                    )
                    .execute()
                )

            msg = _get()
            summaries.append(_normalize_metadata(msg))

        return {"messages": summaries}

    def list_messages(
        self,
        *,
        label_ids: list[str] | None = None,
        q: str | None = None,
        limit: int = 20,
        page_token: str | None = None,
        include_spam_trash: bool = False,
    ) -> dict:
        """
        Lightweight list of message IDs using Gmail's list API.

        Args:
            label_ids: Filter by label IDs
            q: Optional Gmail query string
            limit: Max results (1-100)
            page_token: For pagination
            include_spam_trash: Whether to include spam and trash

        Returns:
            {"messages": [{"id": str, "threadId": str}, ...], "nextPageToken": str | None}
        """
        limit = max(1, min(int(limit or 20), 100))
        svc = self._svc()

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_time=20,
            jitter=None,
            giveup=lambda e: not _is_transient(e),
        )
        def _list():
            req = (
                svc.users()
                .messages()
                .list(
                    userId="me",
                    labelIds=label_ids or [],
                    q=q or "",
                    maxResults=limit,
                    pageToken=page_token,
                    includeSpamTrash=include_spam_trash,
                )
            )
            return req.execute()

        res = _list() or {}
        return {
            "messages": res.get("messages", []),
            "nextPageToken": res.get("nextPageToken"),
        }

    def get_message(self, *, message_id: str) -> dict:
        """
        Fetch a single message (metadata only).

        Args:
            message_id: Gmail message ID

        Returns:
            {"message": MessageSummary}
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
            return (
                svc.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=_metadata_headers(),
                )
                .execute()
            )

        msg = _get()
        return {"message": _normalize_metadata(msg)}

    def get_message_body(
        self,
        *,
        message_id: str,
        prefer: str = "text",  # "text" or "html"
        max_chars: int = 50000,
    ) -> dict:
        """
        Fetch message content and return body plus key headers.

        Args:
            message_id: Gmail message ID
            prefer: "text" or "html" preference
            max_chars: Maximum characters to return

        Returns:
            {
              "id", "threadId", "subject", "from", "to", "cc", "date",
              "body_text": str | None, "body_html": str | None
            }
        """
        prefer = (prefer or "text").lower()
        prefer_text = prefer == "text"
        svc = self._svc()

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_time=20,
            jitter=None,
            giveup=lambda e: not _is_transient(e),
        )
        def _get_full():
            return (
                svc.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

        def _decode(data: str | None) -> str:
            if not data:
                return ""
            try:
                # Gmail uses URL-safe base64 without padding
                padded = data + "=="[(len(data) % 4) :]
                return base64.urlsafe_b64decode(padded.encode()).decode(
                    errors="replace"
                )
            except Exception:
                return ""

        def _extract(payload: dict) -> tuple[str | None, str | None]:
            if not payload:
                return None, None
            mime = payload.get("mimeType", "")
            body = payload.get("body", {})
            data = body.get("data")
            text_val = html_val = None
            if mime == "text/plain" and data:
                text_val = _decode(data)
            elif mime == "text/html" and data:
                html_val = _decode(data)

            parts = payload.get("parts") or []
            for p in parts:
                t_val, h_val = _extract(p)
                text_val = text_val or t_val
                html_val = html_val or h_val
                if text_val and html_val:
                    break
            return text_val, html_val

        full_msg = _get_full()
        meta = _normalize_metadata(full_msg)
        text_body, html_body = _extract(full_msg.get("payload", {}))

        def _limit(s: str | None) -> str | None:
            if s is None:
                return None
            s = s.strip()
            if len(s) > max_chars:
                return s[:max_chars]
            return s

        return {
            **meta,
            "body_text": _limit(text_body) if prefer_text or not html_body else None,
            "body_html": _limit(html_body) if not prefer_text else None,
        }
