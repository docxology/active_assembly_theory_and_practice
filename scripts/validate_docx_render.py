"""Validate the post-render DOCX artifact and write its freshness receipt."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLBACKEND", "Agg")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docx_validation import (  # noqa: E402
    DOCX_RECEIPT_ARTIFACT,
    write_docx_render_receipt,
)


def main() -> int:
    """Write the receipt and return a fail-closed status for the DOCX lane."""

    receipt_path = write_docx_render_receipt(PROJECT_ROOT)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    print("docx_artifact:", PROJECT_ROOT / str(payload["artifact_path"]))
    print("docx_receipt:", PROJECT_ROOT / DOCX_RECEIPT_ARTIFACT)
    print("docx_figures:", payload["embedded_media_count"])
    print("docx_captions:", payload["caption_count"])
    print("docx_valid:", str(payload["valid"]).lower())
    if payload["errors"]:
        for error in payload["errors"]:
            print("docx_error:", error)
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
