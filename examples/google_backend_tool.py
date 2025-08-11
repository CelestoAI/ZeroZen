from zerozen.integrations.google import (
    GmailTool,
    CalendarTool,
    CredentialRecord,
    UserProviderMetadata,
    UserInfo,
)

# Create structured dataclasses from your DB
provider_metadata = UserProviderMetadata(
    refresh_token="xxx",
    scope="openid https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/calendar",
    expires_at=1754842179,
    id_token="xxx",
)

user_info = UserInfo(email="xxx", sub="xxx", email_verified=True)

# Create credentials with matching client_secret
creds = CredentialRecord(
    access_token="xxx",
    user_provider_metadata=provider_metadata,
    user_info=user_info,
    client_id="xxx",
    client_secret="xxx",
)

# Both tools work with same credentials
gmail = GmailTool(creds)
calendar = CalendarTool(creds)
