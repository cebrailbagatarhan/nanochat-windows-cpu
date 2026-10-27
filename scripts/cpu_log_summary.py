"""Read-only summary of CPU training stdout logs; no torch or network required."""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

_NUMBER = r"(\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_STEP = re.compile(rf"\bStep\s+(\d+)\s*/\s*(\d+)\b.*?\bLoss:\s*({_NUMBER})", re.I)
_SECONDS = re.compile(rf"\|\s*({_NUMBER})s(?:/step|\s*\||\s*$)", re.I)
_TRAIN = re.compile(rf"\bTrain(?:\s+loss)?\s*:\s*({_NUMBER})", re.I)
_VAL = re.compile(rf"\bVal(?:\s+loss)?\s*:\s*({_NUMBER})", re.I)
_BEST = re.compile(rf"\bEn iyi val loss\s*:\s*({_NUMBER})", re.I)


def summarize_lines(lines: Iterable[str]) -> dict:
    """Extract observed metrics only; never infer unlogged steps or training success."""
    steps = 0
    last_step = None
    max_steps = None
    last_loss = None
    last_train_loss = None
    last_val_loss = None
    best_val_loss = None
    evaluations = 0
    seconds_total = 0.0
    seconds_samples = 0
    for line in lines:
        step = _STEP.search(line)
        if step:
            steps += 1
            last_step, max_steps = int(step.group(1)), int(step.group(2))
            last_loss = float(step.group(3))
            duration = _SECONDS.search(line)
            if duration:
                seconds_total += float(duration.group(1))
                seconds_samples += 1
        train = _TRAIN.search(line)
        if train:
            last_train_loss = float(train.group(1))
        best = _BEST.search(line)
        val = _VAL.search(line)
        if val and not best:
            last_val_loss = float(val.group(1))
            best_val_loss = min(best_val_loss, last_val_loss) if best_val_loss is not None else last_val_loss
            evaluations += 1
        if best:
            reported = float(best.group(1))
            best_val_loss = min(best_val_loss, reported) if best_val_loss is not None else reported
    if not steps and not evaluations and best_val_loss is None:
        raise ValueError('No CPU training metrics found in the log')
    return {
        'logged_steps': steps,
        'last_step': last_step,
        'max_steps': max_steps,
        'last_training_loss': last_loss,
        'last_evaluation_train_loss': last_train_loss,
        'last_validation_loss': last_val_loss,
        'best_validation_loss': best_val_loss,
        'evaluations': evaluations,
        'average_logged_step_seconds': round(seconds_total / seconds_samples, 3) if seconds_samples else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Summarize an existing CPU training log (read-only).')
    parser.add_argument('log', type=Path, help='Text log captured from a CPU training script')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON')
    args = parser.parse_args(argv)
    try:
        with args.log.open('r', encoding='utf-8', errors='replace') as source:
            result = summarize_lines(source)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            print(f'{key}: {value if value is not None else "not logged"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
