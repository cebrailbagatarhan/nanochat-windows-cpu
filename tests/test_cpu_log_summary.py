"""Dependency-free tests for the CPU training log summarizer."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.cpu_log_summary import main, summarize_lines


class CpuLogSummaryTests(unittest.TestCase):
    def test_fast_log(self):
        result = summarize_lines([
            'Step   0/50 | Loss: 11.0904 | 0.25s/step\n',
            'Step  10/50 | Loss: 7.3542 | 0.20s/step\n',
            '   Train: 7.2000 | Val: 7.5000\n',
            'En iyi val loss: 7.5000\n',
        ])
        self.assertEqual(result['logged_steps'], 2)
        self.assertEqual(result['last_step'], 10)
        self.assertEqual(result['max_steps'], 50)
        self.assertEqual(result['best_validation_loss'], 7.5)
        self.assertEqual(result['evaluations'], 1)
        self.assertEqual(result['average_logged_step_seconds'], 0.225)

    def test_multiline_evaluation(self):
        result = summarize_lines([
            'Step  20/200 | Loss: 4.3 | 1.2s | Kalan: ~5dk\n',
            '   Train loss: 4.2000\n',
            '   Val loss:   4.4000\n',
            '   Val loss:   4.1000\n',
        ])
        self.assertEqual(result['last_evaluation_train_loss'], 4.2)
        self.assertEqual(result['last_validation_loss'], 4.1)
        self.assertEqual(result['best_validation_loss'], 4.1)
        self.assertEqual(result['evaluations'], 2)

    def test_max_format_and_partial_log(self):
        result = summarize_lines(['Step 100/500 (20.0%) | Loss: 5.3 | 2.5s | Kalan: ~1.0saat'])
        self.assertEqual(result['last_step'], 100)
        self.assertIsNone(result['best_validation_loss'])
        self.assertEqual(result['average_logged_step_seconds'], 2.5)

    def test_no_metrics_is_error(self):
        with self.assertRaisesRegex(ValueError, 'No CPU training metrics'):
            summarize_lines(['hello world', 'Val loss: not a number'])

    def test_cli_json_does_not_change_input(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'train.log'
            content = 'Step 0/50 | Loss: 9.2 | 0.5s/step\n'
            file.write_text(content, encoding='utf-8')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(file), '--json']), 0)
            self.assertEqual(json.loads(output.getvalue())['last_training_loss'], 9.2)
            self.assertEqual(file.read_text(encoding='utf-8'), content)


if __name__ == '__main__':
    unittest.main()
