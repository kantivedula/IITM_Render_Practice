# Emily — Living Knowledge Enquiry Agent

Runnable FastAPI application with a small web chat interface, Gemini-backed
training enquiry assistant, optional LangSmith tracing, public website and
optional OneDrive knowledge ingestion, and a proposal preview/accept/download
flow.

## Requirements

- Python 3.10 or newer
- A Google AI API key with access to `gemini-2.5-flash-lite`
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

The web chat asks for an enquiry, then **Prepare initial proposal** creates a
draft using the collected details and configured knowledge. The proposal opens
in a preview page. Its **Accept this draft and enable download** button enables
an HTML download.

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

## Optional OneDrive knowledge

Create a Microsoft Entra app registration with Microsoft Graph **application**
permission to read only the intended knowledge location, grant admin consent,
and set `MS_TENANT_ID`, `MS_CLIENT_ID`, and `MS_CLIENT_SECRET`. Set
`ONEDRIVE_DRIVE_ID` and `ONEDRIVE_FOLDER_PATH` to the approved drive and folder.
Emily reads supported `.txt`, `.md`, `.docx`, and `.pdf` files directly inside
that folder (not subfolders), with a 2 MB per-file limit. If those settings are
blank, only the public website is used.

Only place material approved for proposal drafting in this folder. The app
skips documents whose filenames look like client, case-study, proposal, quote,
pricing, or commercial records, and filters lines that appear sensitive; these
are safeguards, not a substitute for curating the folder or access permissions.

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
to Render; ngrok is disabled in that deployment. In Render, create a new
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
