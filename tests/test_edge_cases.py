import pytest
from app.extraction import extract_text

@pytest.mark.parametrize('filename', ['notes.txt', 'notes.md', 'notes.unknown'])
def test_utf8_bom_is_not_part_of_extracted_text(filename):
    assert extract_text(filename, 'text/plain', b'\xef\xbb\xbfHello') == 'Hello'

def test_plain_utf8_and_noninitial_bom_are_preserved():
    assert extract_text('notes.txt', None, 'Hello\ufeffworld'.encode()) == 'Hello\ufeffworld'
