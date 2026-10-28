from unittest.mock import patch
from app.extraction import extract_text


def test_parameterized_pdf_mime_for_extensionless_upload():
    with patch('app.extraction._extract_pdf_text', return_value='PDF content') as pdf:
        assert extract_text('upload', ' Application/PDF; charset=binary ', b'%PDF') == 'PDF content'
        pdf.assert_called_once_with(b'%PDF')


def test_known_text_extension_still_overrides_pdf_mime():
    assert extract_text('notes.md', 'APPLICATION/PDF; charset=binary', b'hello') == 'hello'
