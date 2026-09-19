import unittest

from scripts.grade_browser_intent import main  # noqa: F401 - import smoke test


class GradeImportTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
