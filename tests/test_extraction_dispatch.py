import unittest
from unittest.mock import patch

from app.extraction import extract_text


class ExtractionDispatchTests(unittest.TestCase):
    def test_known_text_extension_overrides_stale_pdf_mime(self):
        for filename in ('notes.txt', 'README.MD'):
            with self.subTest(filename=filename), patch('app.extraction._extract_pdf_text') as pdf:
                self.assertEqual(extract_text(filename, 'application/pdf', b'hello'), 'hello')
                pdf.assert_not_called()

    def test_pdf_extension_overrides_text_mime(self):
        with patch('app.extraction._extract_pdf_text', return_value='pdf text') as pdf:
            self.assertEqual(extract_text('report.PDF', 'text/plain', b'pdf'), 'pdf text')
            pdf.assert_called_once_with(b'pdf')

    def test_missing_extension_uses_mime(self):
        with patch('app.extraction._extract_pdf_text', return_value='pdf text'):
            self.assertEqual(extract_text('upload', 'application/pdf', b'pdf'), 'pdf text')
        self.assertEqual(extract_text('upload', None, b'hello'), 'hello')
