# src/utils/cache.py
import sqlite3
import os

class SemanticCache:
    def __init__(self, db_path="semantic_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, result TEXT)")
        conn.commit()
        conn.close()

    def get(self, key: str):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT result FROM cache WHERE key=?", (key,))
            row = cursor.fetchone()
            conn.close()
            if row:
                import json
                return json.loads(row[0])
        except:
            pass
        return None

    def set(self, key: str, value: dict):
        try:
            import json
            conn = sqlite3.connect(self.db_path)
            conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            conn.commit()
            conn.close()
        except:
            pass