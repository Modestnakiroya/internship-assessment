"""
Streamlit frontend for the Sunbird assessment pipeline.
Calls the FastAPI backend (see backend/main.py).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

from languages import PIPELINE_TARGET_LANGUAGES

load_dotenv()

_raw_to = (os.environ.get("PIPELINE_HTTP_TIMEOUT") or "").strip()
_PIPELINE_HTTP_TIMEOUT_S = int(_raw_to) if _raw_to else 1800

STEP_LABELS = ("Input", "Transcribe", "Summarise", "Translate", "Speech")


def _backend_url() -> str:
    default = "http://127.0.0.1:8000"
    raw = (os.environ.get("BACKEND_URL") or default).strip().rstrip("/")
    if not raw:
        return default
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return default
    if host.endswith("hf.space") or "huggingface.co" in raw.lower():
        return default
    return raw


def _pipeline_url(backend: str) -> str:
    base = (backend or "").strip().rstrip("/") or "http://127.0.0.1:8000"
    return f"{base}/pipeline"


def post_pipeline(
    backend: str,
    *,
    target_language: str,
    text: str | None = None,
    audio: tuple[str, bytes, str] | None = None,
) -> requests.Response:
    url = _pipeline_url(backend)
    if audio is not None:
        name, raw, mime = audio
        files = {
            "target_language": (None, target_language),
            "audio": (name, raw, mime or "application/octet-stream"),
        }
        return requests.post(url, files=files, timeout=_PIPELINE_HTTP_TIMEOUT_S)

    files = {
        "target_language": (None, target_language),
        "text": (None, (text or "").strip()),
    }
    return requests.post(url, files=files, timeout=_PIPELINE_HTTP_TIMEOUT_S)


def _inject_layout_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

          /* ── Global reset ── */
          html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif !important;
          }

          /* ── Sidebar: hide completely ── */
          section[data-testid="stSidebar"],
          div[data-testid="stSidebar"],
          div[data-testid="stSidebarCollapsedControl"],
          button[data-testid="collapsedControl"] {
            display: none !important;
            width: 0 !important;
          }

          /* ── Header bar ── */
          header[data-testid="stHeader"] {
            background: #0a3d38;
            border-bottom: 1px solid rgba(20,184,166,0.25);
          }

          /* ── Page background ── */
          .stApp {
            background: #f7f6f2;
          }

          /* ── Main content width ── */
          .block-container {
            max-width: 820px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 0 !important;
            padding-bottom: 3rem !important;
          }

          /* ── Hero banner ── */
          .sb-hero-wrap {
            background: #0a3d38;
            margin: -1rem -1rem 2rem -1rem;
            padding: 3rem 2rem 2.5rem 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
          }
          .sb-hero-wrap::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse 80% 60% at 50% 110%, rgba(20,184,166,0.18) 0%, transparent 70%);
          }
          .sb-eyebrow {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #5eead4;
            border: 1px solid rgba(94,234,212,0.3);
            border-radius: 100px;
            padding: 0.3rem 0.85rem;
            margin-bottom: 1.1rem;
          }
          .sb-hero-wrap h1 {
            font-family: 'DM Serif Display', Georgia, serif !important;
            font-size: clamp(2.2rem, 5vw, 3rem) !important;
            font-weight: 400 !important;
            color: #f0faf9 !important;
            line-height: 1.15 !important;
            margin: 0 0 0.75rem 0 !important;
            letter-spacing: -0.01em;
          }
          .sb-hero-wrap h1 em {
            font-style: italic;
            color: #5eead4;
          }
          .sb-hero-sub {
            color: rgba(240,250,249,0.65);
            font-size: 1rem;
            line-height: 1.65;
            max-width: 36rem;
            margin: 0 auto 1.5rem;
            font-weight: 300;
          }
          .sb-pipeline-steps {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 100px;
            padding: 0.45rem 1.1rem;
            font-size: 0.78rem;
            font-weight: 500;
            color: rgba(240,250,249,0.55);
            letter-spacing: 0.04em;
          }
          .sb-pipeline-steps .arrow {
            color: #5eead4;
            opacity: 0.7;
          }
          .sb-pipeline-steps .step-label {
            color: rgba(240,250,249,0.7);
          }

          /* ── Cards / containers ── */
          div[data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #ffffff !important;
            border: 1px solid #e8e5de !important;
            border-radius: 16px !important;
            box-shadow: 0 2px 12px rgba(10,61,56,0.06), 0 1px 3px rgba(10,61,56,0.04) !important;
            padding: 1.5rem !important;
          }

          /* ── Card section titles ── */
          .sb-card-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #0d9488;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #f0ede6;
          }
          .sb-card-title::before {
            content: '';
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #0d9488;
            flex-shrink: 0;
          }

          /* ── Labels / prompts ── */
          .sb-prompt {
            font-size: 0.925rem;
            color: #1c3736;
            margin: 1rem 0 0.4rem 0;
            font-weight: 500;
          }

          /* ── Radio buttons ── */
          div[data-testid="stRadio"] label {
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            color: #374151 !important;
          }
          div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 1rem !important;
          }

          /* ── Text area ── */
          textarea {
            border: 1.5px solid #d4d0c8 !important;
            border-radius: 10px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.95rem !important;
            color: #1a2e2c !important;
            background: #fafaf8 !important;
            transition: border-color 0.18s, box-shadow 0.18s !important;
          }
          textarea:focus {
            border-color: #0d9488 !important;
            box-shadow: 0 0 0 3px rgba(13,148,136,0.12) !important;
            background: #fff !important;
          }

          /* ── Selectbox ── */
          div[data-testid="stSelectbox"] > div > div {
            border: 1.5px solid #d4d0c8 !important;
            border-radius: 10px !important;
            background: #fafaf8 !important;
            font-family: 'DM Sans', sans-serif !important;
          }
          div[data-testid="stSelectbox"] > div > div:focus-within {
            border-color: #0d9488 !important;
            box-shadow: 0 0 0 3px rgba(13,148,136,0.12) !important;
          }

          /* ── File uploader ── */
          div[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #b8e0da !important;
            border-radius: 12px !important;
            background: #f0faf9 !important;
            transition: border-color 0.18s, background 0.18s !important;
          }
          div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #0d9488 !important;
            background: #e6f7f5 !important;
          }

          /* ── Primary button: Run pipeline ── */
          button[kind="primaryFormSubmit"],
          button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #0f766e 0%, #0d9488 60%, #14b8a6 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em !important;
            padding: 0.65rem 1.5rem !important;
            transition: opacity 0.18s, transform 0.12s, box-shadow 0.18s !important;
            box-shadow: 0 4px 14px rgba(13,148,136,0.3) !important;
          }
          button[data-testid="baseButton-primary"]:hover {
            opacity: 0.92 !important;
            box-shadow: 0 6px 20px rgba(13,148,136,0.38) !important;
            transform: translateY(-1px) !important;
          }
          button[data-testid="baseButton-primary"]:active {
            transform: translateY(0) !important;
          }

          /* ── Secondary / download button ── */
          button[data-testid="baseButton-secondary"] {
            border: 1.5px solid #b8e0da !important;
            border-radius: 8px !important;
            color: #0f766e !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            background: #f0faf9 !important;
            transition: background 0.15s, border-color 0.15s !important;
          }
          button[data-testid="baseButton-secondary"]:hover {
            background: #e0f5f2 !important;
            border-color: #0d9488 !important;
          }

          /* ── Result step headings ── */
          .sb-step-out {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #0f766e;
            margin: 1.5rem 0 0.5rem 0;
          }
          .sb-step-out::before {
            content: '';
            display: inline-block;
            width: 20px;
            height: 2px;
            background: #0d9488;
            border-radius: 2px;
            flex-shrink: 0;
          }

          /* ── Result body text ── */
          .sb-result-body {
            font-size: 1rem;
            line-height: 1.7;
            color: #1c3736;
            background: #f7faf9;
            border: 1px solid #dff0ec;
            border-left: 3px solid #0d9488;
            border-radius: 0 8px 8px 0;
            padding: 0.85rem 1rem;
            margin: 0;
          }

          /* ── Timing/meta line ── */
          .sb-meta {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.82rem;
            color: #64748b;
            background: #f1f0eb;
            border-radius: 100px;
            padding: 0.3rem 0.9rem;
            margin-bottom: 1.25rem;
          }
          .sb-meta strong {
            color: #0f766e;
          }

          /* ── Streamlit info / warning / error ── */
          div[data-testid="stAlert"] {
            border-radius: 10px !important;
            border-width: 1px !important;
          }

          /* ── Audio player ── */
          audio {
            width: 100% !important;
            border-radius: 8px !important;
            margin-top: 0.25rem;
          }

          /* ── Caption text ── */
          .stCaption, small {
            font-size: 0.78rem !important;
            color: #94a3b8 !important;
          }

          /* ── Expander ── */
          details summary {
            font-size: 0.85rem !important;
            color: #0f766e !important;
            font-weight: 500 !important;
          }

          /* ── Spinner ── */
          div[data-testid="stSpinner"] p {
            font-size: 0.9rem !important;
            color: #0f766e !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_error(resp: requests.Response) -> str:
    msg: str
    try:
        body: Any = resp.json()
    except ValueError:
        msg = resp.text[:4000] or f"HTTP {resp.status_code}"
        if resp.status_code == 405:
            msg += _format_error_405_hint(resp)
        if getattr(resp, "url", None):
            msg += f"\n\n**Request URL:**\n{resp.url}"
        return msg

    detail = body.get("detail")
    if isinstance(detail, list):
        parts = []
        for err in detail:
            if isinstance(err, dict):
                loc = err.get("loc", ())
                em = err.get("msg", "")
                parts.append(f"{'/'.join(str(x) for x in loc)}: {em}")
            else:
                parts.append(str(err))
        msg = "; ".join(parts) if parts else str(body)
    elif detail is not None:
        msg = str(detail)
    else:
        msg = str(body)

    if resp.status_code == 405:
        msg += _format_error_405_hint(resp)
    if getattr(resp, "url", None):
        msg += f"\n\n**Request URL:**\n{resp.url}"
    return msg


def _format_error_405_hint(resp: requests.Response) -> str:
    allow = resp.headers.get("Allow", "")
    hint = (
        "\n\n**HTTP 405 — Method Not Allowed.** The path exists but this HTTP method is not allowed here. "
        "Typical causes: (1) **`BACKEND_URL` is your public Hugging Face Space URL** (`https://*.hf.space`) — "
        "POST `/pipeline` then hits **Streamlit**, not FastAPI. Use **`http://127.0.0.1:8000`** inside the same "
        "container (see `start.sh`). (2) **`BACKEND_URL` points at Streamlit** (e.g. port **8501**) instead of "
        "**uvicorn** on **8000**. (3) **Stale deploy** — older builds called Sunbird legacy routes (`nllb_translate`, "
        "`/tasks/tts`, RunPod-only STT); redeploy so the client uses `/tasks/translate`, `/tasks/modal/tts`, and "
        "`/tasks/modal/stt`."
    )
    if allow:
        hint += f"\n\n`Allow` response header: `{allow}`"
    return hint


def _hero() -> None:
    steps_html = " <span class='arrow'>→</span> ".join(
        f"<span class='step-label'>{s}</span>" for s in STEP_LABELS
    )
    st.markdown(
        f"""
        <div class="sb-hero-wrap">
          <div class="sb-eyebrow">Sunbird AI</div>
          <h1>Language <em>pipeline</em></h1>
          <p class="sb-hero-sub">
            Summarise your content in clear English, translate it into your chosen language,
            and hear it spoken — powered by the Sunbird API.
          </p>
          <div class="sb-pipeline-steps">{steps_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _card_title(label: str) -> None:
    st.markdown(f'<div class="sb-card-title">{label}</div>', unsafe_allow_html=True)


def _render_output_steps(
    mode: str,
    target_language: str,
    result: dict[str, Any],
    elapsed_s: float,
) -> None:
    st.markdown(
        f'<div class="sb-meta">Completed in <strong>{elapsed_s:.1f}s</strong>'
        f'&nbsp;·&nbsp;target: <strong>{target_language}</strong></div>',
        unsafe_allow_html=True,
    )

    if mode == "Audio" and result.get("transcript"):
        st.markdown(
            f'<div class="sb-step-out">Transcription in {target_language}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="sb-result-body">{result["transcript"]}</p>',
            unsafe_allow_html=True,
        )
    elif mode == "Audio":
        st.markdown('<div class="sb-step-out">Audio transcription</div>', unsafe_allow_html=True)
        st.caption("No transcript was returned for this audio.")

    st.markdown('<div class="sb-step-out">Summary</div>', unsafe_allow_html=True)
    summary_text = result.get("summary", "")
    st.markdown(
        f'<p class="sb-result-body">{summary_text}</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-step-out">Translation</div>', unsafe_allow_html=True)
    translated = result.get("translated_summary", "")
    st.markdown(
        f'<p class="sb-result-body">{translated}</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-step-out">Speech output</div>', unsafe_allow_html=True)
    audio_url = result.get("audio_url")
    if audio_url:
        st.audio(audio_url)
        st.caption("Signed audio URL from Sunbird TTS — play or download soon; it expires after a short time.")
    else:
        st.warning("No audio URL returned.")

    report = []
    if mode == "Audio" and result.get("transcript"):
        report.append("=== Transcription ===\n" + str(result["transcript"]))
    report.append("=== Summary ===\n" + str(result.get("summary", "")))
    report.append("=== Translation ===\n" + str(result.get("translated_summary", "")))
    report.append("=== Audio URL ===\n" + str(result.get("audio_url", "")))
    payload = "\n\n".join(report)
    st.download_button(
        label="⬇  Download report (.txt)",
        data=payload.encode("utf-8"),
        file_name=f"sunbird_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    with st.expander("Raw JSON response"):
        st.json(result)


def main() -> None:
    st.set_page_config(
        page_title="Sunbird AI — Pipeline",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _inject_layout_css()

    backend = _backend_url()
    _hero()

    try:
        input_panel = st.container(border=True)
    except TypeError:
        input_panel = st.container()

    with input_panel:
        _card_title("Input")
        # Form keeps file_uploader value on submit (fixes HF/Streamlit losing the file when a
        # standalone button triggers a rerun before the upload widget state is applied).
        with st.form("sunbird_pipeline_form", clear_on_submit=False):
            mode = st.radio(
                "How would you like to provide content?",
                ["Text", "Audio"],
                horizontal=True,
                key="input_mode",
            )

            target_language = "Luganda"
            text_value = ""
            uploaded = None

            if mode == "Audio":
                st.markdown(
                    '<p class="sb-prompt">Upload your audio file</p>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Recommended: MP3, WAV, OGG, M4A, AAC (max 5 minutes). "
                    "Other formats may still work — the server will reject unsupported audio."
                )
                uploaded = st.file_uploader(
                    "Audio upload",
                    type=None,
                    label_visibility="collapsed",
                    key="pipeline_audio_upload",
                )
                st.markdown(
                    '<p class="sb-prompt">Target language</p>',
                    unsafe_allow_html=True,
                )
                target_language = st.selectbox(
                    "Target language",
                    PIPELINE_TARGET_LANGUAGES,
                    key="target_lang_audio",
                )
            else:
                st.markdown(
                    '<p class="sb-prompt">Enter the text you would like to process</p>',
                    unsafe_allow_html=True,
                )
                text_value = st.text_area(
                    "Text",
                    height=200,
                    placeholder="Paste or type your text here…",
                    label_visibility="collapsed",
                )
                st.markdown(
                    '<p class="sb-prompt">Target language</p>',
                    unsafe_allow_html=True,
                )
                target_language = st.selectbox(
                    "Target language",
                    PIPELINE_TARGET_LANGUAGES,
                    key="target_lang_text",
                )

            submitted = st.form_submit_button(
                "Run pipeline",
                type="primary",
                use_container_width=True,
            )

    try:
        results_panel = st.container(border=True)
    except TypeError:
        results_panel = st.container()

    with results_panel:
        _card_title("Results")

        if not st.session_state.get("last_pipeline_result") and not st.session_state.get(
            "last_pipeline_error"
        ):
            st.info(
                "Fill in the **Input** section above, then press **Run pipeline**."
            )

        if st.session_state.get("last_pipeline_error"):
            st.error(st.session_state["last_pipeline_error"])

        stored = st.session_state.get("last_pipeline_result")
        if stored:
            _render_output_steps(
                str(stored.get("mode", "Text")),
                str(stored.get("target_language", "")),
                stored["result"],
                float(stored.get("elapsed_s", 0)),
            )

    if not submitted:
        return

    if mode == "Text" and not text_value.strip():
        st.warning("Please enter some text before running the pipeline.")
        return
    if mode == "Audio" and uploaded is None:
        st.warning("Please upload an audio file before running the pipeline.")
        return

    st.session_state["last_pipeline_error"] = None
    st.session_state["last_pipeline_result"] = None

    t0 = time.perf_counter()
    with st.spinner("Processing… transcription can take a moment"):
        try:
            resp = post_pipeline(
                backend,
                target_language=target_language,
                text=text_value.strip() if mode == "Text" else None,
                audio=(
                    (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                    if mode == "Audio" and uploaded is not None
                    else None
                ),
            )
        except requests.RequestException as exc:
            st.session_state["last_pipeline_result"] = None
            st.session_state["last_pipeline_error"] = (
                f"Could not reach the backend at `{backend}`: {exc}\n\n"
                "Start the API from the project root: "
                "`uvicorn backend.main:app --reload --port 8000`"
            )
            st.rerun()

    elapsed = time.perf_counter() - t0

    if not resp.ok:
        st.session_state["last_pipeline_result"] = None
        st.session_state["last_pipeline_error"] = _format_error(resp)
        st.rerun()

    result = resp.json()
    st.session_state["last_pipeline_result"] = {
        "result": result,
        "elapsed_s": elapsed,
        "mode": mode,
        "target_language": target_language,
    }
    st.session_state["last_pipeline_error"] = None
    st.rerun()


if __name__ == "__main__":
    main()