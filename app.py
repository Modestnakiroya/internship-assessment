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

STEP_LABELS = ("Input", "Transcribe", "Summarise", "Translate", "Speech")


def _backend_url() -> str:
    """
    FastAPI base URL (not the browser / Streamlit URL).

    On Hugging Face Spaces, users sometimes set BACKEND_URL to the public Space URL
    (https://*.hf.space). POST /pipeline there hits Streamlit, which responds with
    HTTP 405 Method Not Allowed. Always use the internal uvicorn address instead.
    """
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
    """POST /pipeline on the FastAPI root (no trailing slash issues)."""
    base = (backend or "").strip().rstrip("/") or "http://127.0.0.1:8000"
    return f"{base}/pipeline"


def post_pipeline(
    backend: str,
    *,
    target_language: str,
    text: str | None = None,
    audio: tuple[str, bytes, str] | None = None,
) -> requests.Response:
    """
    POST multipart/form-data to /pipeline — matches FastAPI Form + optional File.

    Always use multipart (even for text-only) so ``File(None)`` + ``Form`` routes
    behave consistently across proxies and Starlette versions.
    """
    url = _pipeline_url(backend)
    if audio is not None:
        name, raw, mime = audio
        files = {
            "target_language": (None, target_language),
            "audio": (name, raw, mime or "application/octet-stream"),
        }
        return requests.post(url, files=files, timeout=600)

    files = {
        "target_language": (None, target_language),
        "text": (None, (text or "").strip()),
    }
    return requests.post(url, files=files, timeout=600)


def _inject_layout_css() -> None:
    st.markdown(
        """
        <style>
          /* Hide sidebar completely */
          section[data-testid="stSidebar"],
          div[data-testid="stSidebar"] {
            display: none !important;
            width: 0 !important;
          }
          div[data-testid="stSidebarCollapsedControl"],
          button[data-testid="collapsedControl"] {
            display: none !important;
          }
          header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #0f766e 0%, #0d9488 55%, #14b8a6 100%);
          }
          .block-container {
            max-width: 880px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
          }
          .sb-hero {
            text-align: center;
            padding: 1.25rem 0 0.5rem 0;
          }
          .sb-hero h1 {
            font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin-bottom: 0.35rem;
          }
          .sb-hero p {
            color: #475569;
            font-size: 1.05rem;
            line-height: 1.55;
            max-width: 38rem;
            margin: 0 auto;
          }
          .sb-brand {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #0d9488;
            margin-bottom: 0.35rem;
          }
          .sb-card-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0f766e;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.75rem;
          }
          .sb-prompt {
            font-size: 1rem;
            color: #0f172a;
            margin: 0.75rem 0 0.5rem 0;
            font-weight: 500;
          }
          .sb-step-out {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0f766e;
            margin-top: 1.1rem;
            margin-bottom: 0.35rem;
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
    st.markdown(
        """
        <div class="sb-hero">
          <div class="sb-brand">Sunbird AI</div>
          <h1>Language pipeline</h1>
          <p>
            Summarise your content in clear English, translate it into your chosen language,
            and hear it spoken — powered by the Sunbird API through this assessment backend.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    step_line = " → ".join(STEP_LABELS)
    st.markdown(
        f"<p style='text-align:center;color:#475569;font-size:0.92rem;margin:0.5rem 0 1.25rem 0;"
        f"font-weight:500;letter-spacing:0.02em;'>{step_line}</p>",
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
        f'<p style="color:#64748b;font-size:0.9rem;margin:0 0 1rem 0;">'
        f"Completed in <strong>{elapsed_s:.1f}s</strong> · target: <strong>{target_language}</strong></p>",
        unsafe_allow_html=True,
    )

    if mode == "Audio" and result.get("transcript"):
        st.markdown(
            f'<div class="sb-step-out">Audio transcription text in {target_language}:</div>',
            unsafe_allow_html=True,
        )
        st.write(result["transcript"])
    elif mode == "Audio":
        st.markdown(
            '<div class="sb-step-out">Audio transcription</div>',
            unsafe_allow_html=True,
        )
        st.caption("No transcript was returned for this audio.")

    st.markdown('<div class="sb-step-out">Summary:</div>', unsafe_allow_html=True)
    st.write(result.get("summary", ""))

    st.markdown('<div class="sb-step-out">Translation:</div>', unsafe_allow_html=True)
    st.write(result.get("translated_summary", ""))

    st.markdown('<div class="sb-step-out">Speech output:</div>', unsafe_allow_html=True)
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
        label="Download report (.txt)",
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
                '<p class="sb-prompt">Please provide path to the audio file '
                "(Audio length less than 5 minutes):</p>",
                unsafe_allow_html=True,
            )
            st.caption("Upload your file below (browser security does not allow typing a disk path).")
            uploaded = st.file_uploader(
                "Audio upload",
                type=["mp3", "wav", "ogg", "m4a", "aac"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                st.markdown(
                    '<p class="sb-prompt">Please choose the target language:</p>',
                    unsafe_allow_html=True,
                )
                target_language = st.selectbox(
                    "Target language",
                    PIPELINE_TARGET_LANGUAGES,
                    key="target_lang_audio",
                )
        else:
            st.markdown(
                '<p class="sb-prompt">Please enter the text you would like to process:</p>',
                unsafe_allow_html=True,
            )
            text_value = st.text_area(
                "Text",
                height=200,
                placeholder="Paste or type your text here…",
                label_visibility="collapsed",
            )
            st.markdown(
                '<p class="sb-prompt">Please choose the target language:</p>',
                unsafe_allow_html=True,
            )
            target_language = st.selectbox(
                "Target language",
                PIPELINE_TARGET_LANGUAGES,
                key="target_lang_text",
            )

        run = st.button("Run pipeline", type="primary", use_container_width=True)

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
                "Fill in the **Input** section above, then press **Run pipeline**. "
                "Ensure the FastAPI server is running (`uvicorn backend.main:app --port 8000`)."
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

    if not run:
        return

    st.session_state["last_pipeline_error"] = None
    st.session_state["last_pipeline_result"] = None

    if mode == "Text" and not text_value.strip():
        st.warning("Please enter some text before running the pipeline.")
        return
    if mode == "Audio" and uploaded is None:
        st.warning("Please upload an audio file before running the pipeline.")
        return

    t0 = time.perf_counter()
    with st.spinner("Processing on the server… (transcription can take a while)"):
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
