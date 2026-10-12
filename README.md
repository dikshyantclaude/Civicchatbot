# CivicBuddy

A local Flask app for Nepalese legal-information chat, document questions, appointment requests, document templates, and personal case records.

## Start on this computer

Double-click `start.cmd`, or run `.venv\Scripts\python.exe app.py` from the project folder. Open http://127.0.0.1:5000.

## Install on another computer (Python 3.10+)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-ai.txt
```

Copy `.env.example` to `.env` only if `.env` does not already exist. Set `GROQ_API_KEY`. For the shared law corpus, also set `PINECONE_API_KEY`, `PINECONE_INDEX`, and `PINECONE_NAMESPACE`. Keep `.env` private.

`requirements.txt` installs the basic app and PDF support. `requirements-ai.txt` additionally installs Pinecone and the local embedding model libraries. AI integrations initialize only when needed; other pages work without AI credentials.

## Chat and documents

- Groq generates answers from retrieved source excerpts, with source labels. Default model: `qwen/qwen3.8-27b`; override with `GROQ_MODEL`.
- Corpus retrieval expects a Pinecone index using normalized, 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` vectors and metadata containing `text`, `source`, and optional zero-based `page`.
- Use `python ingest.py path/to/pdfs` to populate an EXISTING correctly dimensioned index. This explicitly writes documents to Pinecone. No documents are uploaded automatically at app startup.
- Upload PDF or UTF-8 TXT in chat (10 MB request limit, 100 PDF pages, 250,000 extracted characters). Scanned PDFs require OCR beforehand. DOC, DOCX and image uploads are not supported.
- The active uploaded document stays attached to follow-up questions until Clear Chat. Uploads remain in the local database; selected text excerpts and recent conversation are sent to Groq when answering. Corpus queries are embedded locally and sent to Pinecone.
- History survives restarts. Buffer mode uses the latest 30 turns; window mode supports 1–30. At most 100 turns are retained per owner; AI context uses the latest 10.
- Voice uses browser speech recognition and synthesis, subject to browser support and permissions. No ElevenLabs/AssemblyAI keys are used. Browser speech services may process audio remotely.
- `/stream_response` is a compatibility alias returning a complete answer; it does not simulate token streaming.

## Accounts, appointments and cases

Guest records belong to the current browser session. Create a local account to keep access from another browser. Passwords are hashed, and records are filtered by the authenticated owner. Creating an account imports the current guest records; signing in opens existing account records. Guest cookies expire after seven days. Local accounts do not verify lawyer credentials.

The workspace supports creating, viewing, deleting and exporting appointment requests and cases. Requests are saved locally only: no lawyer is contacted, no slot is reserved, and no email or payment is sent. Booking attachments are explicitly disabled. Contact the lawyer separately. The old Firebase/demo login and simulated booking success have been replaced.

Case studies and quiz material are inherited, unverified educational examples, not a verified legal database. Document forms generate printable drafts for review. Verify legal content independently before relying on it.

## Local storage and deployment

`instance/civicbuddy.sqlite3` contains accounts, chats, document text and records. `instance/.session-key` keeps sessions valid across restarts unless `SECRET_KEY` is provided. Back up both together. They are excluded from version control. Clearing Chat deletes the current owner's messages and active document. Deleting a record removes it from the workspace database.

The server binds to localhost with debug disabled. This is a local application, not a production legal service. Before public deployment, add managed authentication/account recovery, login rate limiting, professional verification, retention controls, HTTPS, production hosting and real scheduling integrations. Use `COOKIE_SECURE=1` behind HTTPS.

## Test

```powershell
.venv\Scripts\python -m unittest -v test_app
```

Tests use isolated databases and mocked AI responses. Real provider access is separate from these tests.

Provider references: https://console.groq.com/docs/models and https://sdk.pinecone.io/python/how-to/vectors/upsert-and-query.html

