"""
Orchestrates STT → summarise (English) → translate (NLLB) → TTS.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tinytag import TinyTag

from backend.sunbird_client import SunbirdAPIError, SunbirdClient
from languages import LANGUAGE_CODE_BY_NAME, PIPELINE_TARGET_LANGUAGES

MAX_AUDIO_SECONDS = 5 * 60

# POST /tasks/tts speaker IDs (Sunbird docs)
SPEAKER_ID_BY_LANGUAGE: Dict[str, int] = {
    "Acholi": 241,
    "Ateso": 242,
    "Runyankole": 243,
    "Lugbara": 245,
    "Luganda": 248,
    # English: use default Sunbird TTS voice (reads Latin/English text; same default as docs)
    "English": 248,
}


def _audio_duration_seconds(data: bytes, filename: str) -> float:
    suffix = Path(filename).suffix.lower() or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    try:
        info = TinyTag.get(str(path))
    except Exception as exc:
        raise SunbirdAPIError(f"Could not read audio file for duration: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    if info.duration is None:
        raise SunbirdAPIError("Could not read audio duration (unsupported or corrupt file).")
    return float(info.duration)


def _validate_target_language(name: str) -> None:
    if name not in PIPELINE_TARGET_LANGUAGES:
        raise SunbirdAPIError(
            f"Invalid target_language {name!r}. "
            f"Expected one of: {', '.join(PIPELINE_TARGET_LANGUAGES)}."
        )


def run_pipeline(
    client: SunbirdClient,
    *,
    text: Optional[str],
    audio: Optional[Tuple[bytes, str]],
    target_language: str,
) -> Dict[str, Any]:
    """
    Run the full pipeline.

    `audio` is (bytes, filename) when provided. Exactly one of `text` or `audio`.
    """
    _validate_target_language(target_language)

    if (text and text.strip()) and audio:
        raise SunbirdAPIError("Provide either text or audio, not both.")
    if (not text or not str(text).strip()) and not audio:
        raise SunbirdAPIError("Provide either text or audio.")

    transcript: Optional[str] = None
    if audio:
        raw, fname = audio
        duration = _audio_duration_seconds(raw, fname)
        if duration > MAX_AUDIO_SECONDS:
            raise SunbirdAPIError(
                f"Audio is too long ({duration / 60:.1f} minutes). "
                f"Maximum allowed length is 5 minutes."
            )
        lang_code = LANGUAGE_CODE_BY_NAME[target_language]
        transcript = client.transcribe(raw, fname, lang_code)
        source_text = transcript
    else:
        source_text = str(text).strip()

    summary = client.summarise(source_text)
    if target_language == "English":
        translated = summary
    else:
        translated = client.translate_to_ugandan(summary, target_language)
    speaker_id = SPEAKER_ID_BY_LANGUAGE[target_language]
    tts = client.synthesise(translated, speaker_id=speaker_id)

    return {
        "transcript": transcript,
        "summary": summary,
        "translated_summary": translated,
        "audio_url": tts["audio_url"],
        "audio_sample_rate": tts.get("sample_rate"),
        "audio_format": tts.get("format"),
    }
