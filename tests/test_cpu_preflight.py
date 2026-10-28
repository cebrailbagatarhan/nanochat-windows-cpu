"""Tests for the dependency-light CPU readiness inventory."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cpu_preflight


class CPUPreflightTests(unittest.TestCase):
    def test_collect_report_reports_disk_and_cores(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(cpu_preflight.os, "cpu_count", return_value=8):
                report = cpu_preflight.collect_report(Path(folder))
            self.assertEqual(report["cpu_logical_cores"], 8)
            self.assertGreaterEqual(report["disk_free_gib"], 0)
            self.assertIn("torch_installed", report)
            self.assertIn("ram_available_gib", report)

    def test_json_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cpu_preflight.main(["--json", "--workspace", folder]), 0)
            self.assertEqual(json.loads(out.getvalue())["workspace"], str(Path(folder).resolve()))

    def test_disk_threshold_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            for threshold, expected_status in (("5.50", 0), ("5.51", 1)):
                with self.subTest(threshold=threshold):
                    out = io.StringIO()
                    with patch.object(cpu_preflight, "collect_report", return_value={"disk_free_gib": 5.5}):
                        with contextlib.redirect_stdout(out):
                            status = cpu_preflight.main(["--workspace", folder, "--json", "--min-free-gib", threshold])
                    result = json.loads(out.getvalue())
                    self.assertEqual(status, expected_status)
                    self.assertEqual(result["disk_requirement_met"], expected_status == 0)
                    self.assertEqual(result["minimum_disk_free_gib"], float(threshold))

    def test_invalid_disk_threshold_rejected(self):
        for threshold in ("-1", "nan", "inf", "-inf", "invalid"):
            with self.subTest(threshold=threshold):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    cpu_preflight.main(["--min-free-gib", threshold])
                self.assertEqual(error.exception.code, 2)

    def test_missing_workspace_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = str(Path(folder) / "not-here")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cpu_preflight.main(["--workspace", missing])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
