"""Display names and API language codes shared by CLI tools."""

from __future__ import annotations

from typing import Dict, List

# User-facing labels (assessment)
SUPPORTED_LANGUAGES: List[str] = [
    "English",
    "Luganda",
    "Runyankole",
    "Ateso",
    "Lugbara",
    "Acholi",
]

# ISO-style codes used by STT and NLLB translation
LANGUAGE_CODE_BY_NAME: Dict[str, str] = {
    "English": "eng",
    "Luganda": "lug",
    "Runyankole": "nyn",
    "Ateso": "teo",
    "Lugbara": "lgg",
    "Acholi": "ach",
}

# Target languages for translation / TTS in the web app (assessment Part 3)
UGANDAN_TARGET_LANGUAGES: List[str] = [
    "Luganda",
    "Runyankole",
    "Ateso",
    "Lugbara",
    "Acholi",
]
