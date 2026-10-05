# Emily — Living Knowledge Enquiry Agent

Runnable FastAPI application with a small web chat interface, Gemini-backed
training enquiry assistant, optional LangSmith tracing, public website and
optional OneDrive knowledge ingestion, and a proposal preview/accept/download
flow.

## Requirements

- Python 3.10 or newer
- A Google AI API key with access to the configured `GOOGLE_MODEL`
- Google Search grounding enabled for the associated Gemini API project
- Optional LangSmith account/API key for tracing
- Optional Microsoft Entra application and approved OneDrive folder for private
  knowledge

## Run locally

From this folder, in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GOOGLE_API_KEY`. LangSmith tracing is enabled by default
in the example configuration; set `LANGSMITH_API_KEY` and optionally change
`LANGSMITH_PROJECT`. Do not commit `.env` or share its secrets.

Start the server:

```powershell
python agent.py
```

Open <http://127.0.0.1:8000>. The API docs are at
<http://127.0.0.1:8000/docs> and the health endpoint is
<http://127.0.0.1:8000/health>.

The web chat asks for an enquiry, then **Generate proposal** searches supported
documents under the configured OneDrive folder, selects relevant training
content, performs Google Search grounding for the training topic, and drafts a
proposal from those sources and the enquiry. The preview displays the grounded
answer with its citations and Search suggestions in a separate section.
**Accept this draft and enable download** enables an HTML download. Proposal
generation requires a readable OneDrive folder with relevant documents and
Google Search grounding to return cited results; otherwise the page shows an
actionable error.

## Endpoints

- `GET /` — chat page
- `GET /health` — liveness check
- `POST /api/chat` — send `{ "message": "..." }`; include the returned
  `session_id` on later turns
- `POST /api/proposals` — send `{ "session_id": "..." }` to prepare a draft
- `GET /proposals/{proposal_id}/preview?session_id=...` — proposal webpage
- `POST /api/proposals/{proposal_id}/accept` — accept with a JSON
  `{ "session_id": "..." }` body
- `GET /proposals/{proposal_id}/download?session_id=...` — download after
  acceptance

Chat sessions and proposals are stored in a local SQLite database at
`data/emily.sqlite3` by default. Restrict filesystem access and configure
retention/backups before using this with real personal information. When ngrok is enabled, a basic-auth gate protects the entire app using the
configured tunnel username and password. The app still has no individual
accounts, rate limiting, or production data controls; do not use it for real
enquiries until these are added.

## OneDrive proposal knowledge

Create a Microsoft Entra app registration with Microsoft Graph **application**
permission to read only the intended knowledge location, grant admin consent,
and set `MS_TENANT_ID`, `MS_CLIENT_ID`, and `MS_CLIENT_SECRET`. Set
`ONEDRIVE_DRIVE_ID` and `ONEDRIVE_FOLDER_PATH` to the approved drive and folder.
Configure all five values in Render's Environment settings. If one or more are
missing, the proposal action now returns a clear configuration error. Emily
searches supported `.txt`, `.md`, `.docx`, and `.pdf` files in that folder and
its subfolders (up to 40 documents, 20 folders, and 2 MB per file), then selects
text matching the training topic. Files with names indicating proposals,
quotations, pricing, or financial material are excluded, and contact details
and explicitly sensitive lines are redacted.

Use a dedicated folder containing only material approved for proposal drafting.
Do not connect a folder containing other clients' confidential records: the
application has no per-client authentication or OneDrive permission isolation,
and text redaction is not a security boundary.

## LangSmith

LangChain tracing is controlled through the standard environment variables:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=living-knowledge-emily
```

Tracing sends model inputs and outputs to the configured LangSmith project.
Conversation inputs can contain personal information, so review your
organization's privacy requirements and LangSmith data-retention settings
before enabling tracing for real enquiries. Disable tracing with
`LANGSMITH_TRACING=false` when appropriate.

## Google Search grounding

Emily can use Gemini's built-in Google Search grounding to answer in-scope
questions that ask for current or independently verifiable training/OD facts,
research, trends, or examples. The model first creates a short generic search
topic that omits names, organizations, and contact details. Search-grounded
answers display their Google-provided citations and Search suggestions in the
chat. Proposals also use a generic training-topic query for Google Search
grounding after OneDrive retrieval; queries are assembled from training
categories and skills rather than requester names, organizations, or contact
details. The proposal preview and download display
the grounded answer, Google-provided citations, and Search suggestions in a
separate section; the generated proposal content itself does not name its
sources. Gemini's Google Search grounding requires its citations and Search
suggestions to be shown to the user.

Set `GOOGLE_SEARCH_ENABLED=false` to turn this feature off. It is enabled in
`.env.example` and the Render Blueprint. Google Search grounding availability,
quota, and any charges depend on the Gemini API project and model. Google
retains prompts, context, and outputs used for grounding for 30 days under its
Gemini API terms; avoid putting personal or confidential information into
search topics. Review these terms and your data-handling requirements before
enabling it for public users. LangSmith tracing may also capture model
interactions, so review tracing privacy settings separately.

## ngrok tunnel

The app can start an ngrok tunnel automatically when Uvicorn starts and close
it when the app stops. The app listens on `0.0.0.0:8000` by default, and the
tunnel forwards to the local port.

1. Create/sign in to an ngrok account and copy its authtoken from the ngrok
   dashboard.
2. In `.env`, set:

   ```text
   NGROK_ENABLED=true
   NGROK_AUTH_TOKEN=your-ngrok-authtoken
   NGROK_BASIC_AUTH_USER=emily
   NGROK_BASIC_AUTH_PASSWORD=use-a-long-unique-password
   ```

   Keep secrets private; do not commit `.env`.
3. Start the app:

   ```powershell
   python agent.py
   ```

4. Copy the `https://...ngrok...` URL printed in the server log and open it in
   a browser. Enter the configured basic-auth username and password when
   prompted. Open `http://127.0.0.1:8000` for local access (it also requires
   those credentials while the tunnel is enabled).
5. Press `Ctrl+C` to stop Uvicorn; Emily disconnects its tunnel during
   shutdown.

For a reserved/custom ngrok domain, set `NGROK_DOMAIN` to the domain assigned
to your account. Leave it blank to use an automatically assigned ngrok domain.
The `pyngrok` package manages the ngrok agent; on its first use it may download
the ngrok executable.

If startup reports `No module named 'pyngrok'`, install dependencies into the
same virtual environment used to start Emily:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run the app with that interpreter to ensure it uses the same environment:

```powershell
.\.venv\Scripts\python.exe agent.py
```

To use the ngrok CLI manually instead, leave `NGROK_ENABLED=false` and run
`ngrok http 8000` in a second terminal after starting the app. That manual
method does not apply Emily's built-in basic-auth gate, so only use it for
short-lived testing with non-sensitive data.

This remains a development tunnel, not production hosting. Tunnel access is
publicly reachable, and basic auth does not provide per-user authorization,
rate limiting, or hardened data handling.

## Deploy to Render

This repository includes a `render.yaml` Blueprint for deploying Emily directly
to Render; the app detects Render's `RENDER=true` environment variable and
disables ngrok there even if `NGROK_ENABLED=true` is set accidentally. The
Blueprint also sets `NGROK_ENABLED=false`. In Render, create a new
Blueprint instance from the repository and set the prompted `GOOGLE_API_KEY`
secret. Optionally set LangSmith keys and configure **all five** Microsoft
Graph/OneDrive values if using that integration. Leave all OneDrive values
unset to use the public website only. Do not set only some of the OneDrive
values: Emily treats a partial configuration as an error when preparing a
proposal.

The Blueprint uses a paid Starter web service with a 1 GB persistent disk
mounted at `/var/data`; SQLite is configured to use that disk so session and
proposal records survive redeploys. Confirm Render's current plan and disk
pricing before deploying. Render supplies the `PORT` variable, runs Uvicorn,
and checks `/health`.

The Render service is public and the app does not require user login. Treat
this Blueprint as a starting point for deployment, not production-ready
security. Before using real enquiries, add suitable access control, abuse
protection, and a reviewed privacy/retention policy. LangSmith tracing is off
by default in `render.yaml`; enable it only after reviewing what user data is
sent to the tracing service.
