# creds.py
import os
from typing import Callable
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Read-only for search
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def desktop_creds_provider_factory(
    credentials_file: str = "credentials.json",
    token_file: str = "token.json",
) -> Callable[[str], Credentials]:
    """
    Returns a creds_provider(user_id) function that:
    - loads token.json if present,
    - otherwise runs the local browser OAuth flow,
    - refreshes tokens automatically,
    - persists new tokens to token.json.
    """

    def _provider(user_id: str) -> Credentials:
        del user_id  # single-user desktop flow; ignore
        creds = None
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(token_file, "w") as f:
                f.write(creds.to_json())
        return creds

    return _provider
