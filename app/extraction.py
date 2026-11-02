"""Text extraction from uploaded files (plain text or PDF)."""

from __future__ import annotations

import io

SUPPORTED_CONTENT_TYPES = {
    "text/plain",
    "application/pdf",
}


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file's type/extension is not supported."""


def extract_text(filename: str, content_type: str | None, data: bytes) -> str:
    """Extract raw text from an uploaded file.

    Dispatches on file extension (more reliable than the client-supplied
    Content-Type header, which browsers/tools often get wrong or omit).
    """
    lower_name = (filename or "").lower()
    content_type = (content_type or "").split(";", 1)[0].strip().lower()

    if lower_name.endswith(".pdf"):
        return _extract_pdf_text(data)

    if lower_name.endswith((".txt", ".md")):
        return _decode_text(data)

    # Known extensions take precedence over stale or incorrect MIME headers.
    # Consult the header only when the filename does not identify the format.
    if content_type == "application/pdf":
        return _extract_pdf_text(data)

    if (content_type or "").startswith("text/"):
        return _decode_text(data)

    # Fall back to treating it as plain text -- most "unknown" uploads in a
    # demo app are just text files with an unusual extension.
    try:
        return _decode_text(data)
    except UnicodeDecodeError as exc:
        raise UnsupportedFileTypeError(
            f"Unsupported file type for '{filename}'. Upload a .txt, .md, or .pdf file."
        ) from exc


def _decode_text(data: bytes) -> str:
    # UTF-32 LE shares the UTF-16 LE prefix, so inspect longer BOMs first.
    # Only explicit BOMs select a non-UTF-8 encoding.
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        text = data.decode("utf-32")
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig")
    # NUL bytes can decode as UTF-8 but cannot be stored in PostgreSQL text.
    # Check after BOM decoding so valid UTF-16/32 documents remain supported.
    if "\x00" in text:
        raise UnsupportedFileTypeError("Document contains NUL characters; upload a text document.")
    return text


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n\n".join(pages_text)
