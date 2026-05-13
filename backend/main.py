"""
FastAPI routes for transcription, summarisation, translation, TTS, and the full pipeline.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tinytag import TinyTag

from backend.pipeline import MAX_AUDIO_SECONDS, SPEAKER_ID_BY_LANGUAGE, run_pipeline
from backend.sunbird_client import SunbirdAPIError, SunbirdClient
from languages import PIPELINE_TARGET_LANGUAGES

app = FastAPI(title="Sunbird Internship Assessment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client() -> SunbirdClient:
    try:
        return SunbirdClient()
    except SunbirdAPIError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _handle_sunbird(exc: SunbirdAPIError) -> HTTPException:
    msg = str(exc)
    if "too long" in msg.lower() or "Provide either" in msg or "not both" in msg.lower():
        return HTTPException(status_code=400, detail=msg)
    if "Invalid target_language" in msg or "Unsupported" in msg:
        return HTTPException(status_code=400, detail=msg)
    return HTTPException(status_code=502, detail=msg)


def _duration_seconds_or_raise(data: bytes, filename: str) -> float:
    suffix = Path(filename or "clip.mp3").suffix.lower() or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    try:
        info = TinyTag.get(str(path))
    finally:
        path.unlink(missing_ok=True)
    if info.duration is None:
        raise HTTPException(
            status_code=400,
            detail="Could not determine audio duration. Use a supported format.",
        )
    return float(info.duration)


class SummariseBody(BaseModel):
    text: str = Field(..., min_length=1)


class TranslateBody(BaseModel):
    text: str = Field(..., min_length=1)
    target_language: str


class SynthesiseBody(BaseModel):
    text: str = Field(..., min_length=1)
    language: str


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (mp3, wav, ogg, m4a, aac)"),
    language: str = Form(
        "eng",
        description="STT language hint (ISO-style code, e.g. eng, lug, nyn)",
    ),
) -> dict:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    duration = _duration_seconds_or_raise(raw, audio.filename or "clip.mp3")
    if duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Audio is too long ({duration / 60:.1f} minutes). "
                f"Maximum allowed length is 5 minutes."
            ),
        )
    try:
        client = _client()
        text = client.transcribe(raw, audio.filename or "audio.mp3", language)
    except SunbirdAPIError as exc:
        raise _handle_sunbird(exc) from exc
    return {"transcript": text}


@app.post("/summarise")
async def summarise(body: SummariseBody) -> dict:
    try:
        client = _client()
        summary = client.summarise(body.text)
    except SunbirdAPIError as exc:
        raise _handle_sunbird(exc) from exc
    return {"summary": summary}


@app.post("/translate")
async def translate(body: TranslateBody) -> dict:
    if body.target_language not in PIPELINE_TARGET_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"target_language must be one of: {', '.join(PIPELINE_TARGET_LANGUAGES)}.",
        )
    try:
        client = _client()
        translated = client.translate_freeform(body.text, body.target_language)
    except SunbirdAPIError as exc:
        raise _handle_sunbird(exc) from exc
    return {"translation": translated}


@app.post("/synthesise")
async def synthesise(body: SynthesiseBody) -> dict:
    if body.language not in SPEAKER_ID_BY_LANGUAGE:
        raise HTTPException(
            status_code=400,
            detail=f"language must be one of: {', '.join(sorted(SPEAKER_ID_BY_LANGUAGE))}.",
        )
    try:
        client = _client()
        out = client.synthesise(body.text, speaker_id=SPEAKER_ID_BY_LANGUAGE[body.language])
    except SunbirdAPIError as exc:
        raise _handle_sunbird(exc) from exc
    return out


@app.post("/pipeline")
async def pipeline(
    target_language: str = Form(..., description="Ugandan target language name"),
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
) -> dict:
    if target_language not in PIPELINE_TARGET_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"target_language must be one of: {', '.join(PIPELINE_TARGET_LANGUAGES)}.",
        )

    raw_text = (text or "").strip()
    audio_bytes: bytes | None = None
    audio_name: str | None = None
    if audio is not None:
        audio_bytes = await audio.read()
        audio_name = audio.filename or "audio.mp3"
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload.")

    if raw_text and audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Provide either text or audio, not both.",
        )
    if not raw_text and not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Provide either text or audio.",
        )

    try:
        client = _client()
        if audio_bytes:
            pair = (audio_bytes, audio_name or "audio.mp3")
            result = run_pipeline(
                client,
                text=None,
                audio=pair,
                target_language=target_language,
            )
        else:
            result = run_pipeline(
                client,
                text=raw_text,
                audio=None,
                target_language=target_language,
            )
    except SunbirdAPIError as exc:
        raise _handle_sunbird(exc) from exc
    return result


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
