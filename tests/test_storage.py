import json
import sqlite3
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app import db


class Cursor:
    def __init__(self, connection): self.connection=connection
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def executemany(self,query,rows):
        self.connection.executemany(query.replace('%s','?'),[(a,b,c,json.dumps(d)) for a,b,c,d in rows])


class Connection:
    def __init__(self,raw): self.raw=raw
    def transaction(self): return self.raw
    def execute(self,query,values): return self.raw.execute(query.replace('%s','?'),values)
    def cursor(self): return Cursor(self.raw)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.raw=sqlite3.connect(':memory:')
        self.raw.executescript("CREATE TABLE documents(id INTEGER PRIMARY KEY,filename TEXT,content_type TEXT,num_chunks INTEGER); CREATE TABLE chunks(document_id INTEGER,chunk_index INTEGER,content TEXT CHECK(content != 'reject'),embedding TEXT);")
        @contextmanager
        def connection(): yield Connection(self.raw)
        self.patch=patch.object(db,'get_conn',connection);self.patch.start()
    def tearDown(self): self.patch.stop();self.raw.close()
    def test_success_stores_metadata_and_ordered_chunks(self):
        identifier=db.store_document('file.txt','text/plain',['one','two'],[[1.0],[2.0]])
        self.assertEqual(self.raw.execute('SELECT num_chunks FROM documents WHERE id=?',(identifier,)).fetchone(),(2,))
        self.assertEqual(self.raw.execute('SELECT chunk_index,content FROM chunks ORDER BY chunk_index').fetchall(),[(0,'one'),(1,'two')])
    def test_chunk_failure_rolls_back_document_and_all_previous_chunks(self):
        with self.assertRaises(sqlite3.IntegrityError):
            db.store_document('file.txt','text/plain',['one','reject'],[[1.0],[2.0]])
        self.assertEqual(self.raw.execute('SELECT COUNT(*) FROM documents').fetchone(),(0,))
        self.assertEqual(self.raw.execute('SELECT COUNT(*) FROM chunks').fetchone(),(0,))
    def test_embedding_mismatch_writes_nothing(self):
        with self.assertRaises(ValueError): db.store_document('file.txt',None,['one'],[])
        self.assertEqual(self.raw.execute('SELECT COUNT(*) FROM documents').fetchone(),(0,))
