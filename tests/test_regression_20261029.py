import pytest
from app.extraction import extract_text

@pytest.mark.parametrize('codec,bom', [('utf-16-le', b'\xff\xfe'), ('utf-16-be', b'\xfe\xff')])
def test_bom_identified_utf16_uploads_preserve_unicode(codec, bom):
    text = 'Research notes: café and 東京'
    assert extract_text('notes.txt', 'text/plain', bom + text.encode(codec)) == text

def test_binary_input_still_rejected():
    with pytest.raises(UnicodeDecodeError):
        extract_text('notes.txt', 'text/plain', b'\xffinvalid')
