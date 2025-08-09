from rich import print
from gmail_tool import GmailTool
from creds import desktop_creds_provider_factory


def main():
    creds_provider = desktop_creds_provider_factory(
        credentials_file="credentials.json",  # downloaded from Google Console
        token_file="token.json",
    )
    gmail = GmailTool(creds_provider)

    # Example: unread Stripe invoices from last 30 days
    res = gmail.search_messages(
        user_id="local",  # ignored by desktop provider
        query="from:google newer_than:30d",
        limit=10,
    )

    for m in res["messages"]:
        print(f"{m['date']} | {m['from']} | {m['subject']} | {m['snippet']}")
        print("\n")


if __name__ == "__main__":
    main()
