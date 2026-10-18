import unittest
from app.chunking import chunk_text

class ChunkParameterTests(unittest.TestCase):
    def test_invalid_parameters_fail_for_empty_and_nonempty_text(self):
        for text in ['', 'hello world']:
            for params in [{'chunk_size': 0}, {'chunk_size': 2.5}, {'chunk_size': True}, {'overlap': -1}, {'overlap': 1.5}, {'overlap': False}]:
                with self.subTest(text=text, params=params), self.assertRaises(ValueError):
                    chunk_text(text, **params)

    def test_valid_sliding_window(self):
        self.assertEqual(chunk_text('abcdef', 4, 2), ['abcd', 'cdef'])
        self.assertEqual(chunk_text('', 4, 0), [])
