"""Dependency-free tests; no downloads, model loading or PyTorch needed."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_profile import main, profile_dataset


class DatasetProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "shakespeare.txt"

    def test_character_and_line_counts(self):
        self.path.write_text("ab\ncd\n", encoding="utf-8")
        result = profile_dataset(self.path, 0.5)
        self.assertEqual(result["characters"], 6)
        self.assertEqual(result["lines"], 2)
        self.assertEqual(result["unique_characters"], 5)
        self.assertEqual(result["train_characters"], 3)
        self.assertEqual(result["validation_characters"], 3)

    def test_unicode_bytes_differ_from_characters(self):
        self.path.write_text("ş🙂", encoding="utf-8")
        result = profile_dataset(self.path)
        self.assertEqual(result["characters"], 2)
        self.assertEqual(result["bytes"], 6)
        self.assertEqual(result["lines"], 1)

    def test_empty_file_rejected(self):
        self.path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            profile_dataset(self.path)

    def test_invalid_fraction_rejected(self):
        self.path.write_text("hello", encoding="utf-8")
        for fraction in (0, 1, -0.2, 1.2, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                profile_dataset(self.path, fraction)

    def test_missing_file_rejected(self):
        with self.assertRaises(FileNotFoundError):
            profile_dataset(self.path)

    def test_json_cli_no_raw_content(self):
        self.path.write_text("SECRET SAMPLE", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--path", str(self.path), "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["characters"], 13)
        self.assertNotIn("SECRET SAMPLE", output.getvalue())

    def test_cli_error_code(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--path", str(self.path)]), 2)

    def test_blank_and_repeated_lines_without_disclosing_text(self):
        self.path.write_text("Alpha\\r\\n  \\r\\nAlpha\\r\\n Alpha \\r\\nBeta\\r\\n\\r\\n", encoding="utf-8")
        result = profile_dataset(self.path)
        self.assertEqual(result["blank_lines"], 2)
        self.assertEqual(result["duplicate_nonempty_lines"], 2)
        self.assertNotIn("Alpha", str(result))

    def test_repeated_lines_are_case_sensitive(self):
        self.path.write_text("Alpha\\nalpha\\nAlpha\\n", encoding="utf-8")
        result = profile_dataset(self.path)
        self.assertEqual(result["duplicate_nonempty_lines"], 1)
        self.assertEqual(result["blank_lines"], 0)

    def test_blank_and_duplicate_metrics_in_json_cli(self):
        self.path.write_text("a\\na\\n \\n", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--path", str(self.path), "--json"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["duplicate_nonempty_lines"], 1)
        self.assertEqual(result["blank_lines"], 1)


if __name__ == "__main__":
    unittest.main()
