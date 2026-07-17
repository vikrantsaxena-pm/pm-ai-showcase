"""Download an RBI PDF into ./data for use with the RAG app.

RBI's website is often behind a bot/CAPTCHA check, so a direct download may
return an HTML challenge page instead of a PDF. This script verifies the bytes
are actually a PDF and fails loudly (rather than saving a CAPTCHA page as .pdf)
so you can fall back to downloading in your browser and using the app's uploader.

Usage:
    python fetch_sample.py <pdf-url> [output-name.pdf]

Find circulars at https://www.rbi.org.in (Notifications / Master Directions).
Example (e-mandate / recurring transactions notification page):
    https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12051
Open it, copy the PDF link, and pass it here — or just download and upload it.
"""

from __future__ import annotations

import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch(url: str, out_name: str | None = None) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_name = out_name or os.path.basename(url.split("?")[0]) or "rbi_document.pdf"
    if not out_name.lower().endswith(".pdf"):
        out_name += ".pdf"
    dest = os.path.join(DATA_DIR, out_name)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (rbi-rag-sample)"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - user-supplied URL
        data = resp.read()

    if not data.startswith(b"%PDF"):
        raise SystemExit(
            "The downloaded content is not a PDF (likely an HTML/CAPTCHA page).\n"
            "Open the URL in your browser, download the PDF, and upload it in the app instead."
        )

    with open(dest, "wb") as fh:
        fh.write(data)
    print(f"Saved {len(data):,} bytes -> {dest}")
    return dest


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    fetch(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
