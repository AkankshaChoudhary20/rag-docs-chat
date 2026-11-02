import pytest
from app.extraction import extract_text, UnsupportedFileTypeError

@pytest.mark.parametrize('name,mime', [('binary.bin', None), ('claimed.txt', 'text/plain')])
def test_nul_containing_binary_is_rejected(name, mime):
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(name, mime, b'hello\x00world')

@pytest.mark.parametrize('encoding', ['utf-16', 'utf-32'])
def test_bom_encoded_text_is_decoded_before_binary_check(encoding):
    assert extract_text('note.txt', None, 'hello'.encode(encoding)) == 'hello'
