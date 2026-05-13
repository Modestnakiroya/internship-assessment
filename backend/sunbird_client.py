"""
Thin HTTP client for Sunbird AI tasks used by the assessment pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from languages import LANGUAGE_CODE_BY_NAME


class SunbirdAPIError(Exception):
    """Raised when the Sunbird API returns an unexpected or error payload."""


def _response_json(resp: requests.Response) -> Any:
    """Parse JSON from a successful Sunbird response; raise SunbirdAPIError on garbage."""
    try:
        return resp.json()
    except ValueError as exc:
        snippet = (resp.text or "")[:800]
        raise SunbirdAPIError(
            f"Sunbird API returned non-JSON (HTTP {resp.status_code}). "
            f"Body starts with: {snippet!r}"
        ) from exc


def _post(url: str, **kwargs: Any) -> requests.Response:
    try:
        return requests.post(url, **kwargs)
    except requests.RequestException as exc:
        raise SunbirdAPIError(f"Network error calling Sunbird API: {exc}") from exc


class SunbirdClient:
    BASE = "https://api.sunbird.ai"

    def __init__(self, token: Optional[str] = None) -> None:
        load_dotenv()
        self._token = (token or os.environ.get("SUNBIRD_API_TOKEN", "")).strip()
        if not self._token:
            raise SunbirdAPIError(
                "SUNBIRD_API_TOKEN is not set. Add it to your environment or .env file."
            )

    def _headers_json(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _headers_auth_only(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str,
    ) -> str:
        """POST /tasks/stt — multipart upload."""
        url = f"{self.BASE}/tasks/stt"
        files = {
            "audio": (filename or "audio.mp3", audio_bytes, "application/octet-stream"),
        }
        data = {
            "language": language_code,
            "adapter": language_code,
            "recognise_speakers": "false",
            "whisper": "false",
        }
        resp = _post(
            url,
            headers=self._headers_auth_only(),
            files=files,
            data=data,
            timeout=300,
        )
        self._raise_for_bad_response(resp)
        payload = _response_json(resp)
        text = _extract_stt_text(payload)
        if not text:
            raise SunbirdAPIError(f"No transcription in API response: {payload!r}")
        return text

    def summarise(self, text: str) -> str:
        """Summarise via Sunflower simple (English summary)."""
        instruction = (
            "Summarize the following text in clear, concise English "
            "(2 to 5 sentences). Do not add a title or preamble. "
            "Output only the summary.\n\n"
            f"{text}"
        )
        return self._sunflower_simple(instruction)

    def translate_to_ugandan(self, text: str, target_language_name: str) -> str:
        """Translate English text into a Ugandan language using NLLB."""
        if target_language_name not in LANGUAGE_CODE_BY_NAME:
            raise SunbirdAPIError(f"Unsupported target language: {target_language_name!r}")
        tgt = LANGUAGE_CODE_BY_NAME[target_language_name]
        src = LANGUAGE_CODE_BY_NAME["English"]
        url = f"{self.BASE}/tasks/nllb_translate"
        resp = _post(
            url,
            json={"text": text, "source_language": src, "target_language": tgt},
            headers=self._headers_json(),
            timeout=120,
        )
        self._raise_for_bad_response(resp)
        data = _response_json(resp)
        status = str(data.get("status", "")).lower()
        output = data.get("output") or {}
        err = output.get("Error")
        if err:
            raise SunbirdAPIError(str(err))
        if status and status != "success":
            raise SunbirdAPIError(f"Translation failed with status={data.get('status')!r}")
        translated = output.get("translated_text")
        if not translated:
            raise SunbirdAPIError(f"No translated_text in response: {data!r}")
        return str(translated)

    def translate_freeform(self, text: str, target_language_name: str) -> str:
        """Translate arbitrary text to a target language (Sunflower)."""
        instruction = (
            f"Translate the following text into {target_language_name}. "
            f"Reply with only the translated text — no notes, labels, or quotes.\n\n"
            f"{text}"
        )
        return self._sunflower_simple(instruction)

    def synthesise(self, text: str, speaker_id: int) -> Dict[str, Any]:
        """POST /tasks/tts — returns dict including audio_url."""
        url = f"{self.BASE}/tasks/tts"
        resp = _post(
            url,
            json={"text": text, "speaker_id": speaker_id, "temperature": 0.6},
            headers=self._headers_json(),
            timeout=180,
        )
        self._raise_for_bad_response(resp)
        data = _response_json(resp)
        out = data.get("output") or {}
        audio_url = out.get("audio_url") or data.get("audio_url")
        if not audio_url:
            raise SunbirdAPIError(f"No audio_url in TTS response: {data!r}")
        return {
            "audio_url": audio_url,
            "sample_rate": out.get("sample_rate"),
            "format": out.get("format", "mp3"),
            "duration_seconds": out.get("duration_seconds"),
        }

    def _sunflower_simple(self, instruction: str) -> str:
        url = f"{self.BASE}/tasks/sunflower_simple"
        resp = _post(
            url,
            data={
                "instruction": instruction,
                "model_type": "qwen",
                "temperature": "0.25",
            },
            headers=self._headers_auth_only(),
            timeout=180,
        )
        self._raise_for_bad_response(resp)
        data = _response_json(resp)
        if data.get("success") is False:
            raise SunbirdAPIError(f"Sunflower inference failed: {data!r}")
        content = data.get("response")
        if not content and isinstance(data.get("output"), dict):
            out = data["output"]
            content = out.get("response") or out.get("content") or out.get("text")
        if not content:
            raise SunbirdAPIError(f"No response field in Sunflower payload: {data!r}")
        return str(content).strip()

    @staticmethod
    def _raise_for_bad_response(resp: requests.Response) -> None:
        if resp.ok:
            return
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text[:2000]
        raise SunbirdAPIError(f"HTTP {resp.status_code}: {detail}")


def _extract_stt_text(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    if payload.get("audio_transcription"):
        return str(payload["audio_transcription"])
    out = payload.get("output")
    if isinstance(out, dict):
        if out.get("text"):
            return str(out["text"])
        if out.get("audio_transcription"):
            return str(out["audio_transcription"])
    return None
