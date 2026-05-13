---
title: Sunbird
emoji: 🐦
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
---
# Sunbird AI internship assessment — GenAI pipeline app

This project is a small **Streamlit + FastAPI** application that runs a fixed pipeline over **Sunbird AI** only: **Speech-to-Text (optional) → summarise → translate → Text-to-Speech**. You can also call each step through separate REST endpoints for testing.

## Project description

Users provide **either** pasted text **or** an audio file (up to **5 minutes**), choose one of five **Ugandan target languages** (Luganda, Runyankole, Ateso, Lugbara, Acholi), and receive an **English summary**, a **translated summary** in the chosen language, and **spoken audio** of that translation. When the input is audio, the **transcript** is shown as an intermediate step. All model calls go to **https://api.sunbird.ai** (STT, Sunflower summarisation, NLLB translation for the pipeline step, and TTS).

## Architecture

```text
┌─────────────┐     HTTP      ┌──────────────────┐      HTTPS       ┌─────────────────┐
│  Streamlit  │ ────────────► │  FastAPI         │ ────────────────► │  Sunbird AI API │
│   app.py    │   multipart   │  backend/main.py │   Bearer token   │  tasks/*        │
└─────────────┘               └────────┬─────────┘                  └─────────────────┘
                                       │
                              backend/pipeline.py
                              backend/sunbird_client.py
```

- **`app.py`** — Streamlit UI: input mode (text vs audio), target language, calls `POST /pipeline`, shows transcript (if audio), summary, translated summary, and `st.audio` for the signed TTS URL. Surfaces API/validation errors with `st.error`.
- **`backend/main.py`** — FastAPI routes: `/transcribe`, `/summarise`, `/translate`, `/synthesise`, `/pipeline`, plus `/health`.
- **`backend/sunbird_client.py`** — HTTP wrappers: `POST /tasks/stt`, `POST /tasks/sunflower_inference` (summarise + free-form translate), `POST /tasks/nllb_translate` (pipeline translate from English summary), `POST /tasks/tts`.
- **`backend/pipeline.py`** — Orchestrates the end-to-end flow and enforces the **5-minute** audio cap using `tinytag`.

### Pipeline mapping to Sunbird endpoints

| Step | Sunbird endpoint |
|------|------------------|
| STT (audio only) | [`POST /tasks/stt`](https://docs.sunbird.ai/guides/speech-to-text) |
| Summarise | [`POST /tasks/sunflower_inference`](https://docs.sunbird.ai/api-reference/ai/sunflower-chat) |
| Translate (inside `/pipeline`) | [`POST /tasks/nllb_translate`](https://docs.sunbird.ai/) (English summary → local language) |
| Standalone `/translate` | [`POST /tasks/sunflower_inference`](https://docs.sunbird.ai/api-reference/ai/sunflower-chat) (any source text → target language) |
| TTS | [`POST /tasks/tts`](https://docs.sunbird.ai/guides/text-to-speech) |

## Local setup

**Prerequisites:** Python **3.10+**, `git`, and a Sunbird API token from [api.sunbird.ai](https://api.sunbird.ai/).

1. **Clone** your fork and enter the project directory.

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows (cmd):
   venv\Scripts\activate.bat
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables**

   Copy `.env.example` to `.env` and set at least `SUNBIRD_API_TOKEN`.

   ```bash
   cp .env.example .env
   # Edit .env — never commit real tokens.
   ```

5. **Run the backend** (from the project root, with the venv active):

   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Run the Streamlit app** (second terminal, same venv, project root):

   ```bash
   streamlit run app.py
   ```

   Open the URL Streamlit prints (usually `http://localhost:8501`). The UI calls the API at `BACKEND_URL` (default `http://127.0.0.1:8000`).

7. **Part 1 tests** (optional):

   ```bash
   pytest
   ```

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `SUNBIRD_API_TOKEN` | **Yes** (for API routes) | Bearer token for `https://api.sunbird.ai`. |
| `BACKEND_URL` | No | Base URL of FastAPI for `app.py` (default `http://127.0.0.1:8000`). |

## Usage walkthrough

1. Start **uvicorn** and **streamlit** as above and ensure `.env` contains a valid `SUNBIRD_API_TOKEN`.
2. In the browser, choose **Text** or **Audio**.
3. Pick a **target Ugandan language**.
4. Click **Run pipeline**.
5. Review **Transcript** (audio only), **Summary (English)**, **Translated summary**, and play the **synthesised speech** (temporary signed URL from Sunbird TTS).
6. If something fails (network, quota, validation), the error message from the backend is shown in the UI.

### CLI tools (Parts 2a–2b)

- `python translate.py` — interactive translation CLI.
- `python transcribe.py` — interactive STT CLI with the same 5-minute rule.

## API quick reference

All routes expect the server process to have `SUNBIRD_API_TOKEN` set.

| Method | Path | Body |
|--------|------|------|
| `POST` | `/transcribe` | multipart: `audio` (file), optional form `language` (default `eng`) |
| `POST` | `/summarise` | JSON `{"text": "..."}` |
| `POST` | `/translate` | JSON `{"text": "...", "target_language": "Luganda"}` (name must be one of the five locals) |
| `POST` | `/synthesise` | JSON `{"text": "...", "language": "Luganda"}` |
| `POST` | `/pipeline` | multipart: `target_language` (required), optional `text`, optional `audio` file — exactly one of `text` or `audio` |
| `GET` | `/health` | Liveness check |

## Deployment on Hugging Face Spaces

**Important:** A default **Streamlit-only** Space runs a single process. This project also needs **FastAPI** on another port on the same machine, so the most reliable approach is a **Docker** Space (or any host where you can run two processes).

### Docker Space (recommended)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choose **Docker** as the SDK.
2. Push this repository (including `Dockerfile` and `start.sh`). The image starts **FastAPI** on `127.0.0.1:8000`, waits until `GET /health` succeeds, then starts **Streamlit** on port **7860** (configurable).
3. In Space **Settings → Variables and secrets**, add **`SUNBIRD_API_TOKEN`** (required) with your Sunbird bearer token.
   - **Do not** set **`BACKEND_URL`** to your public Space URL (`https://*.hf.space`). From inside the container, Streamlit must call FastAPI at **`http://127.0.0.1:8000`**. The included **`start.sh`** forces that URL before starting Streamlit so a mistaken secret cannot break the app.
   - Optional: **`STREAMLIT_SERVER_PORT`** if your Space expects a different Streamlit port (default `7860`).

### HTTP 405 (`Method Not Allowed`) on Hugging Face

If the UI shows **HTTP 405** for `/pipeline`, Streamlit is almost certainly posting to the **wrong host** (the public `hf.space` URL or the Streamlit port). The API lives only on **`127.0.0.1:8000`** inside the Docker container. Remove a wrong **`BACKEND_URL`** secret, redeploy, or rely on **`start.sh`** which forces the internal URL.

### Space card (README on Hugging Face)

Hugging Face can read YAML **front matter** at the very top of the Space `README.md` for the card title, emoji, and SDK. If your GitHub `README.md` should stay plain for reviewers, add this only on the branch you push to Hugging Face, or prepend when creating the Space:

```yaml
---
title: Sunbird Pipeline
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

Then keep the rest of your project README below the closing `---`.

### Build and run Docker locally

From the project root (Docker installed):

```bash
docker build -t sunbird-pipeline .
docker run --rm -p 7860:7860 -e SUNBIRD_API_TOKEN="your_token_here" sunbird-pipeline
```

Open `http://localhost:7860`. The UI uses `BACKEND_URL=http://127.0.0.1:8000` inside the container by default.

### Streamlit SDK Space (not recommended here)

The built-in Streamlit template only starts Streamlit; it will **not** start FastAPI unless you switch to Docker or host the API on a **separate** public service and set **`BACKEND_URL`** to that API’s base URL (not the Streamlit Space URL).

## Known limitations

- Audio longer than **5 minutes** is rejected locally with a clear error (before calling STT where possible).
- **TTS signed URLs** expire quickly (Sunbird documentation: on the order of minutes). Play or download soon after generation.
- The **pipeline** assumes the **summary** is in **English** so **NLLB** can translate into the selected Ugandan language. Very short or noisy input may yield poor summaries or empty STT.
- Sunbird **rate limits** and occasional **503** responses apply per their docs; the UI surfaces error text from the backend.

## Repository layout

```text
.
├── app.py                  # Streamlit frontend
├── backend/
│   ├── main.py             # FastAPI routes
│   ├── sunbird_client.py   # Sunbird HTTP client
│   └── pipeline.py         # STT → summarise → translate → TTS
├── exercises/              # Part 1
├── tests/
├── translate.py
├── transcribe.py
├── languages.py
├── requirements.txt
├── .env.example
├── Dockerfile
├── start.sh
├── .dockerignore
└── README.md
```

## Licence / attribution

Built for the Sunbird AI internship assessment; API usage is subject to [Sunbird AI](https://sunbird.ai/) terms and quotas.
