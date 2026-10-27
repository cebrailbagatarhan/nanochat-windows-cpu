"""Offline, read-only character profile of a local Shakespeare text file.

This reports character counts, not tokenizer-dependent token counts.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DATASET = Path.home() / ".cache" / "nanochat" / "shakespeare" / "shakespeare.txt"


def profile_dataset(path, train_fraction=0.9):
    """Return aggregate text statistics without exposing dataset contents."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("Dataset is empty")
    total = len(text)
    train_chars = int(total * train_fraction)
    return {
        "bytes": path.stat().st_size,
        "characters": total,
        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
        "unique_characters": len(set(text)),
        "train_characters": train_chars,
        "validation_characters": total - train_chars,
        "train_fraction": train_fraction,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_DATASET, help="Local UTF-8 dataset path")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Character split (default 0.9)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable aggregate statistics")
    args = parser.parse_args(argv)
    try:
        result = profile_dataset(args.path, args.train_fraction)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Dataset profile error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
