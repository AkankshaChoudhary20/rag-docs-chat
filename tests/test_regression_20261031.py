import unittest
from app.extraction import extract_text

class Utf32ExtractionTests(unittest.TestCase):
    def test_utf32_with_bom_decodes_before_utf16_detection(self):
        text = 'Meeting notes: café and 日本語'
        for encoding, bom in [('utf-32-le', b'\xff\xfe\x00\x00'), ('utf-32-be', b'\x00\x00\xfe\xff')]:
            with self.subTest(encoding=encoding):
                self.assertEqual(extract_text('notes.txt', 'text/plain', bom + text.encode(encoding)), text)

    def test_existing_unicode_encodings_remain_supported(self):
        for encoding in ('utf-8-sig', 'utf-16', 'utf-8'):
            self.assertEqual(extract_text('notes.md', None, 'café'.encode(encoding)), 'café')
