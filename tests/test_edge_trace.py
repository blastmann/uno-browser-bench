import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.extract_edge_trace_local import build_prefix_rows, read_sessions, transition_name


class EdgeTracePrivacyTests(unittest.TestCase):
    def test_transition_low_byte(self):
        self.assertEqual(transition_name(1), "typed")
        self.assertEqual(transition_name(0x101), "typed")

    def test_extracts_category_only_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "History"
            db = sqlite3.connect(history)
            db.executescript(
                """
                CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT NOT NULL);
                CREATE TABLE visits (id INTEGER PRIMARY KEY, url INTEGER NOT NULL,
                                     visit_time INTEGER NOT NULL, transition INTEGER NOT NULL);
                """
            )
            db.executemany(
                "INSERT INTO urls(id, url) VALUES (?, ?)",
                [(1, "https://mail.example.test/inbox?secret=1"),
                 (2, "https://github.com/private/repo"),
                 (3, "https://google.com/search?q=secret")],
            )
            base = 13300000000000000
            db.executemany(
                "INSERT INTO visits(id, url, visit_time, transition) VALUES (?, ?, ?, ?)",
                [(1, 1, base, 1), (2, 2, base + 1_000_000, 0),
                 (3, 3, base + 2_000_000, 0)],
            )
            db.commit()
            db.close()

            sessions, stats = read_sessions(history)
            rows, audit = build_prefix_rows(sessions)
            self.assertEqual(stats["raw_rows_scanned"], 3)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(len(rows), 2)
            payload = json.dumps(rows, ensure_ascii=False)
            self.assertNotIn("secret", payload)
            self.assertNotIn("example.test", payload)
            self.assertFalse(audit["privacy"]["raw_urls_written"])
            self.assertEqual(rows[0]["label"]["page_type"], "github_code")


if __name__ == "__main__":
    unittest.main()
