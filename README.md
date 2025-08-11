<center>

<div align="center">

<img src="/assets/zerozen-min.png" alt="zerozen" width="200" />

<br>

# Your hyper-personal, always-on, open-source AI companion.

**Zero friction. Maximum zen. 🧘‍♀️**

Dedicated to the [creators of Zero](https://www.open.ac.uk/blogs/MathEd/index.php/2022/08/25/the-men-who-invented-zero/) — Aryabhatta and Bhaskara ✨

<img src="/assets/cli.png" alt="zerozen CLI" width="500" />

[![PyPI version](https://badge.fury.io/py/zerozen.svg)](https://badge.fury.io/py/zerozen)
[![License: MIT](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/Apache-2)

</div>

</center>

______________________________________________________________________

## ✨ What is ZeroZen?

ZeroZen is your **personal AI assistant** that seamlessly integrates with your daily tools. Think of it as having a smart conversation with your Gmail, Google Calendar, and more — all through a beautiful command-line interface.

**🎯 Perfect for:**

- **Busy professionals** who live in their inbox
- **Developers** who want AI that actually helps with real work
- **Power users** seeking zero-friction automation
- **Privacy-conscious individuals** who prefer local, controllable AI

______________________________________________________________________

## 🚀 Quick Start

### Installation

```bash
pip install zerozen
```

### Setup Google Integration (2 minutes)

1. **Get Google credentials** (one-time setup):

   ```bash
   # Opens setup guide with direct links
   zen setup-google
   ```

1. **Start chatting with your data**:

   ```bash
   zen chat
   ```

1. **Ask questions like**:

   - *"Show me emails from GitHub about security alerts"*
   - *"What meetings do I have this week?"*
   - *"Find invoices from Stripe in my Gmail"*

That's it! 🎉

______________________________________________________________________

## 🛠️ Installation & Setup

### Option 1: Quick Install

```bash
pip install zerozen
zen setup-google  # One-time setup
zen chat          # Start using!
```

### Option 2: From Source

```bash
git clone https://github.com/aniketmaurya/zerozen.git
cd zerozen
pip install -e .
zen setup-google
```

### Google Setup Details

The `zen setup-google` command guides you through:

1. **Creating Google Cloud Project** (if needed)
1. **Enabling APIs** (Gmail, Calendar)
1. **OAuth credentials** (desktop app)
1. **Browser authentication** (automatic)
1. **Credential storage** (secure, local)

**First run:**

```bash
zen setup-google
# ✅ Opens browser for one-time authentication
# ✅ Saves credentials locally
# ✅ Ready to use!
```

**Already set up:**

```bash
zen setup-google
# ✅ Google credentials already exist
# Use --force to re-authenticate
```

______________________________________________________________________

## 💬 Chat Interface

The heart of ZeroZen is its conversational interface. Just talk to your data naturally.

```bash
zen chat
```

### Example Conversations

**📧 Email Management:**

```
You: Find emails from stripe with invoices
AI: 🔍 Found 3 invoices from Stripe in the last 30 days:
    • Dec 15: Monthly subscription - $29.00
    • Nov 15: Pro upgrade - $99.00  
    • Oct 15: Monthly subscription - $29.00
```

**📅 Calendar Queries:**

```
You: What's my schedule tomorrow?
AI: 📅 Tomorrow (Dec 16):
    • 9:00 AM - Team standup (30 min)
    • 2:00 PM - Client presentation (1 hour)
    • 4:30 PM - 1:1 with Sarah (30 min)
```

**🔗 Cross-tool Intelligence:**

```
You: Do I have any meetings about the project mentioned in John's email?
AI: 🔍 Found John's email about "Project Alpha" from yesterday.
    📅 Yes! You have "Project Alpha Planning" tomorrow at 10 AM.
```

### Chat Features

- **🧠 Smart context** - Remembers your conversation
- **🛠️ Multiple tools** - Gmail, Calendar, web search (coming soon)
- **⚡ Fast responses** - Optimized for quick queries
- **🎨 Rich formatting** - Beautiful, readable output
- **💾 Session memory** - Continues where you left off

______________________________________________________________________

## 🧑‍💻 Developer Experience

### Python API

Use ZeroZen programmatically in your own applications:

```python
from zerozen import agents

# Simple agent usage
result = agents.run_sync(
    "Find emails from GitHub about security issues", tools=["search_gmail"], max_turns=3
)
print(result)

# Advanced usage with specific tools
result = agents.run_sync(
    "What's my schedule conflicts next week?",
    tools=["list_calendar_events", "search_gmail"],
    model="gpt-4o",  # Optional model override
)
```

### Custom Integrations

```python
from zerozen.integrations.google import GmailTool, load_user_credentials

# Load your saved credentials
creds = load_user_credentials("credentials.my_google_account.json")

# Direct tool usage
gmail = GmailTool(creds)
messages = gmail.search_messages(query="from:github.com", limit=10)

# Your custom logic here...
```

### Backend Integration

Perfect for server applications:

```python
from zerozen.integrations.google import CredentialRecord, UserProviderMetadata, UserInfo

# Create from your database/API
user_creds = CredentialRecord(
    access_token="ya29.xxx",
    user_provider_metadata=UserProviderMetadata(
        refresh_token="1//xxx",
        scope="gmail.readonly calendar.readonly",
        expires_at=1234567890,
    ),
    user_info=UserInfo(email="user@example.com", sub="google_user_id"),
    client_id="your_oauth_client_id",
    client_secret="your_oauth_secret",
)

# Use with any tool
gmail = GmailTool(user_creds)
```

______________________________________________________________________

## 🔧 Configuration

### CLI Options

```bash
zen chat --help
# Gmail-specific user context
zen chat --gmail-user-id me

# Model selection  
zen chat --model gpt-4o

# Show tool outputs
zen chat --show-tools
```

### Environment Variables

```bash
# Optional: Set default model
export OPENAI_MODEL=gpt-4o

# Optional: Custom credential paths
zen setup-google --credentials-file /path/to/creds.json
zen setup-google --user-storage /path/to/user-creds.json
```

______________________________________________________________________

## 🔐 Security & Privacy

**🛡️ Your data stays yours:**

- **Local credentials** - Stored securely on your machine
- **No data collection** - We don't see your emails or calendar
- **Open source** - Audit the code yourself
- **Standard OAuth** - Uses Google's official authentication

**🔒 Credential management:**

- Automatic token refresh
- Secure local storage
- Per-user isolation
- Configurable file paths

______________________________________________________________________

## 🛣️ Roadmap

| Feature | Status | Description |
|---------|--------|-------------|
| Gmail Integration | ✅ | Search, read, analyze emails |
| Google Calendar | ✅ | View events, check availability |
| Chat Interface | ✅ | Conversational AI with memory |
| Desktop OAuth | ✅ | One-command authentication |
| Backend API | ✅ | Programmatic access |
| **Email Actions** | 🔜 | Draft, reply, send emails |
| **Calendar Management** | 🔜 | Create, update events |
| **Slack Integration** | 🔜 | Team communication |
| **Document AI** | 🔜 | Google Docs, Sheets analysis |
| **Multi-user Support** | 🔜 | Team deployments |
| **Plugin System** | 🔮 | Custom integrations |

______________________________________________________________________

## 🤝 Contributing

We'd love your help making ZeroZen even better!

### Quick Contribution Guide

1. **🍴 Fork & Clone**

   ```bash
   git clone https://github.com/yourusername/zerozen.git
   cd zerozen
   ```

1. **🔧 Development Setup**

   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

   # Install in development mode
   pip install -e .

   # Install development dependencies
   pip install pytest ruff black
   ```

1. **🧪 Run Tests**

   ```bash
   pytest tests/
   ```

1. **🎯 Make Changes & Test**

   ```bash
   # Test your changes
   zen setup-google
   zen chat

   # Run linters
   ruff check .
   black .
   ```

1. **📝 Submit PR**

### Areas We Need Help

- **🔌 New integrations** (Slack, Linear, Notion)
- **🎨 UI improvements** (better formatting, themes)
- **📚 Documentation** (examples, guides)
- **🧪 Testing** (edge cases, error handling)
- **🌍 Internationalization** (non-English support)

______________________________________________________________________

## 📖 More Examples

### Gmail Power Queries

```bash
zen chat
```

```
# Search patterns
You: "Emails from my manager in the last week"
You: "Unread emails with attachments"  
You: "Messages about the Q4 planning"

# Analysis
You: "Summarize the latest email from accounting"
You: "What action items do I have from recent emails?"

# Time-based
You: "Show me emails I received while I was on vacation"
You: "Important emails from the last 3 days"
```

### Calendar Intelligence

```
# Schedule awareness
You: "When am I free for a 1-hour meeting this week?"
You: "Do I have conflicts with the team all-hands?"

# Event analysis  
You: "How many meetings do I have this week?"
You: "What's the longest meeting on my calendar?"

# Cross-reference
You: "Find emails about meetings I have tomorrow"
```

______________________________________________________________________

## 🙏 Acknowledgments

**Built with love using:**

- [OpenAI Agents](https://github.com/openai/agents) - The backbone of our AI system
- [Typer](https://typer.tiangolo.com/) - Beautiful CLI interfaces
- [Rich](https://rich.readthedocs.io/) - Rich text and formatting
- [Google APIs](https://developers.google.com/) - Gmail and Calendar integration

**Special thanks to:**

- The open-source community for inspiration and contributions
- Early beta testers for valuable feedback

______________________________________________________________________

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

______________________________________________________________________

**Ready to experience zero-friction AI?**

```bash
pip install zerozen
zen setup-google
zen chat
```

*Welcome to your **Zen** mode.* 🧘‍♀️✨
