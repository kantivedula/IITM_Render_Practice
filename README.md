# Emily — Living Knowledge Enquiry Agent

Emily is a FastAPI-based learning and training enquiry assistant. It uses
Gemini for training-related conversations and program-outline generation,
Google Search grounding for research, optional LangSmith tracing, and a
browser-based review and approval flow with PDF export.

## Requirements

- Python 3.10 or newer
- A Google AI API key with access to the configured `GOOGLE_MODEL`
- Google Search grounding enabled for the associated Gemini API project
- Optional LangSmith account/API key for tracing

## Run locally

In PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`. LangSmith tracing settings are also available
there; review the data-handling implications before enabling tracing with real
enquiries.

Start the server:

```powershell
python agent.py
```

Open <http://127.0.0.1:8000>. API documentation is available at
<http://127.0.0.1:8000/docs>, and the health endpoint is
<http://127.0.0.1:8000/health>.

## Program-outline workflow

The page collects a training category, intervention type, delivery choice,
participant-count range (`0-10`, `10-25`, `25-50`, or `>50`), participant
hierarchy level, and optional context. Emily uses those details to generate a preliminary program outline and displays
it in the page. The requester can review it, add refinement instructions and
regenerate it, or approve it. The PDF download becomes available only after
approval. Emily does not access OneDrive or other private document stores for
proposal generation.

## API endpoints

- `GET /` — Emily web interface
- `GET /health` — liveness check
- `POST /api/chat` — chat; send `message` and include the returned `session_id`
  for subsequent messages
- `POST /api/proposals` — create an outline from the structured form
- `GET /proposals/{proposal_id}/preview?session_id=...` — view the outline
- `POST /api/proposals/{proposal_id}/regenerate` — regenerate with refinement
  instructions and a `session_id`
- `POST /api/proposals/{proposal_id}/accept` — approve using a `session_id`
- `GET /proposals/{proposal_id}/download?session_id=...` — download the
  approved outline as PDF

Chat sessions and proposals are stored in SQLite at `data/emily.sqlite3` by
default. Configure retention, backups, and filesystem access before using the
application with personal information.

## Google Search grounding

Emily can use Gemini's built-in Google Search grounding to answer in-scope
questions. Search queries are assembled from training information and exclude
requester names and contact details. The chat UI displays Google-provided
citations and Search suggestions alongside grounded answers. Program outlines
are generated directly from the submitted requirements and do not display or
reveal research sources.

Set `GOOGLE_SEARCH_ENABLED=false` to disable grounding. Availability, quotas,
charges, and data handling depend on the Gemini API project and Google's
applicable terms. Avoid entering confidential information into public-facing
enquiries.

## LangSmith

LangChain tracing uses the standard environment variables:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=living-knowledge-emily
```

Tracing may capture model inputs and outputs. Review your privacy requirements
and LangSmith retention settings before enabling it for real enquiries.

## ngrok tunnel

For local testing, set `NGROK_ENABLED=true` and configure
`NGROK_AUTH_TOKEN` in `.env`. Optionally set `NGROK_DOMAIN` for a reserved
domain. The app starts and stops the tunnel with Uvicorn and applies the
configured basic-auth gate while the tunnel is enabled. Render deployment
disables ngrok; use Render's public service URL instead.

## Deploy to Render

The `render.yaml` Blueprint installs dependencies, starts Uvicorn using
Render's `PORT`, checks `/health`, and mounts persistent storage for SQLite.
Set the prompted `GOOGLE_API_KEY` secret in Render. LangSmith settings are
optional. Ngrok is disabled in the Blueprint.

The Render service is publicly reachable and does not provide user accounts.
Add appropriate authentication, abuse protection, and a reviewed
privacy/retention policy before using it with real enquiries.
