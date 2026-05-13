"""
Thin HTTP client for Sunbird AI tasks used by the assessment pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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


# Very long inputs slow Sunflower more than they help the pipeline; cap for latency.
_SUMMARY_INPUT_SOFT_CAP = 12_000


class SunbirdClient:
    BASE = "https://api.sunbird.ai"

    def __init__(self, token: Optional[str] = None) -> None:
        load_dotenv()
        self._token = (token or os.environ.get("SUNBIRD_API_TOKEN", "")).strip()
        if not self._token:
            raise SunbirdAPIError(
                "SUNBIRD_API_TOKEN is not set. Add it to your environment or .env file."
            )
        self._session = requests.Session()

    def _headers_json(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _headers_auth_only(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _post(self, url: str, **kwargs: Any) -> requests.Response:
        try:
            return self._session.post(url, **kwargs)
        except requests.RequestException as exc:
            raise SunbirdAPIError(f"Network error calling Sunbird API: {exc}") from exc

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str,
    ) -> str:
        """POST /tasks/modal/stt — Modal Whisper STT (recommended in Sunbird docs)."""
        url = f"{self.BASE}/tasks/modal/stt"
        files = {
            "audio": (filename or "audio.mp3", audio_bytes, "application/octet-stream"),
        }
        data = {"language": language_code} if language_code else {}
        resp = self._post(
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
        """Summarise via Sunflower chat inference (JSON API)."""
        body = text.strip()
        if len(body) > _SUMMARY_INPUT_SOFT_CAP:
            body = body[:_SUMMARY_INPUT_SOFT_CAP].rsplit(" ", 1)[0] + " …"
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Summarize in clear English in at most 3 short sentences. "
                    "No title or preamble — summary text only."
                ),
            },
            {"role": "user", "content": body},
        ]
        return self._sunflower_inference(messages, temperature=0.2)

    def translate_to_ugandan(self, text: str, target_language_name: str) -> str:
        """Translate English text into a Ugandan language using NLLB."""
        if target_language_name not in LANGUAGE_CODE_BY_NAME:
            raise SunbirdAPIError(f"Unsupported target language: {target_language_name!r}")
        tgt = LANGUAGE_CODE_BY_NAME[target_language_name]
        src = LANGUAGE_CODE_BY_NAME["English"]
        url = f"{self.BASE}/tasks/translate"
        resp = self._post(
            url,
            json={"text": text, "source_language": src, "target_language": tgt},
            headers=self._headers_json(),
            timeout=120,
        )
        self._raise_for_bad_response(resp)
        data = _response_json(resp)
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        err = output.get("Error") if output else None
        if err:
            raise SunbirdAPIError(str(err))
        translated = (
            (output.get("translated_text") if output else None)
            or data.get("translated_text")
            or data.get("translation")
        )
        if translated:
            return str(translated).strip()
        status = str(data.get("status") or "").lower()
        if status in ("error", "failed", "failure"):
            raise SunbirdAPIError(f"Translation failed with status={data.get('status')!r}")
        raise SunbirdAPIError(f"No translation text in response: {data!r}")

    def translate_freeform(self, text: str, target_language_name: str) -> str:
        """Translate arbitrary text to a target language (Sunflower chat inference)."""
        user = (
            f"Translate the following text into {target_language_name}. "
            f"Reply with only the translated text — no notes, labels, or quotes.\n\n"
            f"{text}"
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": user},
        ]
        return self._sunflower_inference(messages, temperature=0.15)

    def synthesise(self, text: str, speaker_id: int) -> Dict[str, Any]:
        """POST /tasks/modal/tts — signed URL mode (Sunbird docs)."""
        url = f"{self.BASE}/tasks/modal/tts"
        resp = self._post(
            url,
            json={
                "text": text,
                "speaker_id": speaker_id,
                "response_mode": "url",
            },
            headers=self._headers_json(),
            timeout=180,
        )
        self._raise_for_bad_response(resp)
        data = _response_json(resp)
        out = data.get("output") if isinstance(data.get("output"), dict) else {}
        audio_url = data.get("audio_url") or out.get("audio_url")
        if not audio_url:
            raise SunbirdAPIError(f"No audio_url in TTS response: {data!r}")
        return {
            "audio_url": audio_url,
            "sample_rate": out.get("sample_rate"),
            "format": out.get("format", "wav"),
            "duration_seconds": out.get("duration_seconds") or data.get("duration_estimate_seconds"),
        }

    def _sunflower_inference(self, messages: List[Dict[str, str]], *, temperature: float) -> str:
        """POST /tasks/sunflower_inference — JSON chat completions (replaces deprecated form-only simple)."""
        url = f"{self.BASE}/tasks/sunflower_inference"
        payload: Dict[str, Any] = {
            "messages": messages,
            "model_type": "qwen",
            "temperature": temperature,
            "stream": False,
        }
        resp = self._post(url, json=payload, headers=self._headers_json(), timeout=180)
        self._raise_for_bad_response(resp)
        data = _response_json(resp)
        content = data.get("content")
        if not content and isinstance(data.get("output"), dict):
            content = data["output"].get("content")
        if not content:
            raise SunbirdAPIError(f"No content in Sunflower inference response: {data!r}")
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
