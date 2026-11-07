from types import SimpleNamespace
from unittest.mock import patch
import pytest
from app.extraction import extract_text, UnsupportedFileTypeError

def test_pdf_nul_is_rejected_before_postgres_ingestion():
    for content in ['hello\x00world', '\x00']:
        reader = lambda stream: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: content)])
        with patch.dict('sys.modules', {'pypdf': SimpleNamespace(PdfReader=reader)}):
            with pytest.raises(UnsupportedFileTypeError, match='NUL'):
                extract_text('report.pdf', 'application/pdf', b'pdf')

def test_pdf_page_boundaries_are_preserved():
    reader = lambda stream: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: 'hello'), SimpleNamespace(extract_text=lambda: 'world')])
    with patch.dict('sys.modules', {'pypdf': SimpleNamespace(PdfReader=reader)}):
        assert extract_text('report.pdf', None, b'pdf') == 'hello\n\nworld'
