"""
CLI: translate text between supported languages using the Sunbird AI API.

Uses NLLB when translating to/from English; otherwise Sunflower simple inference.
Token: SUNBIRD_API_TOKEN in .env (never hardcode).
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import requests
from dotenv import load_dotenv

from languages import LANGUAGE_CODE_BY_NAME, SUPPORTED_LANGUAGES

API_BASE = "https://api.sunbird.ai"
TRANSLATE_URL = f"{API_BASE}/tasks/translate"
SUNFLOWER_INFERENCE_URL = f"{API_BASE}/tasks/sunflower_inference"


def _load_token() -> str:
    load_dotenv()
    token = os.environ.get("SUNBIRD_API_TOKEN", "").strip()
    if not token:
        print(
            "Error: SUNBIRD_API_TOKEN is not set. Add it to a .env file in this directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _prompt_language(label: str) -> str:
    print(f"\n{label} — choose a number:")
    for i, name in enumerate(SUPPORTED_LANGUAGES, start=1):
        print(f"  {i}. {name}")
    while True:
        raw = input("Enter number: ").strip()
        if not raw.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(raw)
        if 1 <= idx <= len(SUPPORTED_LANGUAGES):
            return SUPPORTED_LANGUAGES[idx - 1]
        print(f"Please enter a number between 1 and {len(SUPPORTED_LANGUAGES)}.")


def _translate_nllb(
    token: str, text: str, source_code: str, target_code: str
) -> str:
    headers = {**_auth_headers(token), "Content-Type": "application/json"}
    resp = requests.post(
        TRANSLATE_URL,
        json={
            "text": text,
            "source_language": source_code,
            "target_language": target_code,
        },
        headers=headers,
        timeout=120,
    )
    if not resp.ok:
        _print_http_error("Translation (NLLB)", resp)
        sys.exit(1)
    data = resp.json()
    output = data.get("output") or {}
    err = output.get("Error")
    if err:
        print(f"Translation error: {err}", file=sys.stderr)
        sys.exit(1)
    translated = output.get("translated_text") or data.get("translated_text") or data.get("translation")
    if translated:
        return str(translated).strip()
    status = data.get("status")
    if status and str(status).lower() in ("error", "failed", "failure"):
        print(f"Translation failed: status={status!r}", file=sys.stderr)
        sys.exit(1)
    print("Unexpected API response (no translation text).", file=sys.stderr)
    print(data, file=sys.stderr)
    sys.exit(1)


def _translate_sunflower(
    token: str, text: str, source_name: str, target_name: str
) -> str:
    user = (
        f"Translate the following text from {source_name} to {target_name}. "
        f"Reply with only the translated text — no explanations, labels, or quotes.\n\n"
        f"{text}"
    )
    headers = {**_auth_headers(token), "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": user},
        ],
        "model_type": "qwen",
        "temperature": 0.2,
        "stream": False,
    }
    resp = requests.post(
        SUNFLOWER_INFERENCE_URL,
        json=payload,
        headers=headers,
        timeout=180,
    )
    if not resp.ok:
        _print_http_error("Translation (Sunflower)", resp)
        sys.exit(1)
    data = resp.json()
    content = data.get("content")
    if not content and isinstance(data.get("output"), dict):
        content = data["output"].get("content")
    if not content:
        print("Unexpected API response (no content).", file=sys.stderr)
        print(data, file=sys.stderr)
        sys.exit(1)
    return str(content).strip()


def _print_http_error(context: str, resp: requests.Response) -> None:
    print(f"{context} request failed: HTTP {resp.status_code}", file=sys.stderr)
    try:
        detail = resp.json()
    except ValueError:
        print(resp.text[:2000], file=sys.stderr)
        return
    if isinstance(detail, dict):
        if "detail" in detail:
            print(detail["detail"], file=sys.stderr)
        else:
            print(detail, file=sys.stderr)
    else:
        print(detail, file=sys.stderr)


def main() -> None:
    token = _load_token()
    source = _prompt_language("Source language")
    target = _prompt_language("Target language")
    if source == target:
        print("Error: target language must differ from source language.", file=sys.stderr)
        sys.exit(1)
    text = input("\nText to translate:\n").strip()
    if not text:
        print("Error: empty text.", file=sys.stderr)
        sys.exit(1)

    src_code = LANGUAGE_CODE_BY_NAME[source]
    tgt_code = LANGUAGE_CODE_BY_NAME[target]
    english = LANGUAGE_CODE_BY_NAME["English"]

    if english in (src_code, tgt_code):
        result = _translate_nllb(token, text, src_code, tgt_code)
    else:
        result = _translate_sunflower(token, text, source, target)

    print("\n--- Translation ---\n")
    print(result)


if __name__ == "__main__":
    main()
