import unittest

from scripts.analyze_edge_history import category, edge_time


class EdgeHistoryPrivacyTests(unittest.TestCase):
    def test_categories_are_coarse(self):
        self.assertEqual(category("github.com"), "github_code")
        self.assertEqual(category("mail.example.test"), "email")
        self.assertEqual(category("private.example.test"), "other")

    def test_webkit_timestamp_is_iso(self):
        self.assertTrue(edge_time(11644473600000000).startswith("1970-01-01T00:00:00"))


if __name__ == "__main__":
    unittest.main()
