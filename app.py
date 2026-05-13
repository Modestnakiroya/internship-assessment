"""
Streamlit frontend for the Sunbird assessment pipeline.
Calls the FastAPI backend (see backend/main.py).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from languages import UGANDAN_TARGET_LANGUAGES

load_dotenv()


def _page_shell() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_error(resp: requests.Response) -> str:
    try:
        body: Any = resp.json()
    except ValueError:
        return resp.text[:4000] or f"HTTP {resp.status_code}"
    detail = body.get("detail")
    if isinstance(detail, list):
        parts = []
        for err in detail:
            if isinstance(err, dict):
                loc = err.get("loc", ())
                msg = err.get("msg", "")
                parts.append(f"{'/'.join(str(x) for x in loc)}: {msg}")
            else:
                parts.append(str(err))
        return "; ".join(parts) if parts else str(body)
    if detail is not None:
        return str(detail)
    return str(body)


def _default_backend() -> str:
    return os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _sidebar_connection() -> str:
    st.sidebar.header("Connection")
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = _default_backend()

    backend = st.sidebar.text_input(
        "Backend base URL",
        value=st.session_state.backend_url,
        help="FastAPI root, e.g. http://127.0.0.1:8000 when running locally.",
    ).strip().rstrip("/")
    st.session_state.backend_url = backend or _default_backend()

    b = st.session_state.backend_url
    cols = st.sidebar.columns(2)
    with cols[0]:
        if st.button("Ping /health", use_container_width=True):
            try:
                r = requests.get(f"{b}/health", timeout=5)
                if r.ok:
                    st.session_state["health_ok"] = True
                    st.session_state["health_msg"] = r.text
                else:
                    st.session_state["health_ok"] = False
                    st.session_state["health_msg"] = _format_error(r)
            except requests.RequestException as exc:
                st.session_state["health_ok"] = False
                st.session_state["health_msg"] = str(exc)

    with cols[1]:
        if hasattr(st, "link_button"):
            st.link_button("OpenAPI docs", f"{b}/docs", use_container_width=True)
        else:
            st.markdown(f"[OpenAPI docs]({b}/docs)")

    if "health_ok" in st.session_state:
        if st.session_state["health_ok"]:
            st.sidebar.success("API reachable")
        else:
            st.sidebar.error(st.session_state.get("health_msg", "Unreachable"))

    st.sidebar.divider()
    st.sidebar.subheader("About this UI")
    st.sidebar.markdown(
        "**Pipeline:** optional STT → English summary → translate → TTS. "
        "Audio must be **≤ 5 minutes** (MP3, WAV, OGG, M4A, AAC)."
    )
    st.sidebar.caption("Powered by Sunbird AI via your FastAPI backend.")

    if st.sidebar.button("Clear last run"):
        for key in ("last_pipeline_result", "last_pipeline_error", "health_ok", "health_msg"):
            st.session_state.pop(key, None)
        st.rerun()

    return st.session_state.backend_url


def _render_results(
    backend: str,
    result: dict[str, Any],
    elapsed_s: float,
    mode: str,
    target_language: str,
) -> None:
    st.divider()
    top = st.columns(4)
    top[0].metric("Total time", f"{elapsed_s:.1f}s")
    top[1].metric("Target language", target_language or "—")
    summary = result.get("summary") or ""
    top[2].metric("Summary words", str(len(summary.split())) if summary else "0")
    trans = result.get("translated_summary") or ""
    top[3].metric("Translation words", str(len(trans.split())) if trans else "0")

    tab1, tab2 = st.tabs(["Pipeline output", "Raw JSON"])
    with tab1:
        if mode == "Audio" and result.get("transcript"):
            with st.expander("Transcript", expanded=True):
                st.write(result["transcript"])
        elif mode == "Audio":
            with st.expander("Transcript", expanded=False):
                st.caption("No transcript field returned (unexpected for audio input).")

        with st.expander("Summary (English)", expanded=True):
            st.write(result.get("summary", ""))

        with st.expander("Translated summary", expanded=True):
            st.write(result.get("translated_summary", ""))

        audio_url = result.get("audio_url")
        with st.expander("Synthesised speech", expanded=True):
            if audio_url:
                st.audio(audio_url)
                st.caption(
                    "Signed URL from Sunbird TTS — open or play promptly; it expires after a short time."
                )
            else:
                st.warning("No audio URL returned.")

        report = []
        if mode == "Audio" and result.get("transcript"):
            report.append("=== Transcript ===\n" + str(result["transcript"]))
        report.append("=== Summary (English) ===\n" + str(result.get("summary", "")))
        report.append("=== Translated summary ===\n" + str(result.get("translated_summary", "")))
        report.append("=== Audio URL ===\n" + str(result.get("audio_url", "")))
        payload = "\n\n".join(report)
        st.download_button(
            label="Download report (.txt)",
            data=payload.encode("utf-8"),
            file_name=f"sunbird_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    with tab2:
        st.json(result)

    st.caption(f"Backend used: `{backend}`")


def main() -> None:
    st.set_page_config(
        page_title="Sunbird Pipeline",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _page_shell()

    backend = _sidebar_connection()

    st.title("Sunbird AI pipeline")
    st.markdown(
        "Run **text or audio** through **summarise → translate → speech**, "
        "with intermediate steps visible below."
    )

    left, right = st.columns((1, 1.05), gap="large")

    with left:
        try:
            panel = st.container(border=True)
        except TypeError:
            panel = st.container()
        with panel:
            st.subheader("Input")
            mode = st.radio("Input type", ["Text", "Audio"], horizontal=True, key="input_mode")
            target_language = st.selectbox(
                "Target Ugandan language",
                UGANDAN_TARGET_LANGUAGES,
                help="Translation and TTS voice follow this choice.",
            )

            text_value = ""
            uploaded = None
            if mode == "Text":
                text_value = st.text_area(
                    "Text",
                    height=220,
                    placeholder="Paste or type the content you want summarised and translated…",
                )
            else:
                uploaded = st.file_uploader(
                    "Audio file (max 5 minutes)",
                    type=["mp3", "wav", "ogg", "m4a", "aac"],
                    help="Formats supported by the backend and Sunbird STT.",
                )

            run = st.button("Run full pipeline", type="primary", use_container_width=True)

    with right:
        st.subheader("Results")
        if not st.session_state.get("last_pipeline_result") and not st.session_state.get("last_pipeline_error"):
            st.info(
                "Configure input on the left, then **Run full pipeline**. "
                "Use **Ping /health** in the sidebar if the page cannot reach the API."
            )

        if st.session_state.get("last_pipeline_error"):
            st.error(st.session_state["last_pipeline_error"])

        stored = st.session_state.get("last_pipeline_result")
        if stored:
            st.caption(
                f"Last run: {stored.get('finished_at', '')} · mode: {stored.get('mode', '')}"
            )
            _render_results(
                stored.get("backend", backend),
                stored["result"],
                float(stored.get("elapsed_s", 0)),
                str(stored.get("mode", "Text")),
                str(stored.get("target_language", "")),
            )

    if not run:
        return

    st.session_state["last_pipeline_error"] = None
    st.session_state["last_pipeline_result"] = None

    if mode == "Text" and not text_value.strip():
        st.warning("Please enter some text before running.")
        return
    if mode == "Audio" and uploaded is None:
        st.warning("Please upload an audio file before running.")
        return

    data = {"target_language": target_language}
    files = None
    if mode == "Text":
        data["text"] = text_value.strip()
    else:
        assert uploaded is not None
        files = {
            "audio": (
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type or "application/octet-stream",
            ),
        }

    t0 = time.perf_counter()
    with st.spinner("Running pipeline on the server (STT may take a while)…"):
        try:
            resp = requests.post(
                f"{backend}/pipeline",
                data=data,
                files=files,
                timeout=600,
            )
        except requests.RequestException as exc:
            st.session_state["last_pipeline_result"] = None
            msg = (
                f"Could not reach backend at `{backend}`: {exc}\n\n"
                "Start the API from the project root: "
                "`uvicorn backend.main:app --reload --port 8000`"
            )
            st.session_state["last_pipeline_error"] = msg
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
        "backend": backend,
        "target_language": target_language,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    st.session_state["last_pipeline_error"] = None
    st.rerun()


if __name__ == "__main__":
    main()
