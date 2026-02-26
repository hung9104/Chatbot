from pathlib import Path

from decouple import config

from .theme import Chatbot as ChatbotTheme

PDFJS_VERSION_DIST: str = config("PDFJS_VERSION_DIST", "pdfjs-4.0.379-dist")
PDFJS_PREBUILT_DIR: Path = config(
    "PDFJS_PREBUILT_DIR", Path(__file__).parent / "prebuilt" / PDFJS_VERSION_DIST
)

__all__ = ["ChatbotTheme", "PDFJS_VERSION_DIST", "PDFJS_PREBUILT_DIR"]
