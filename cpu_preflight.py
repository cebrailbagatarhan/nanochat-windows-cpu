"""Read-only CPU training readiness report; does not load PyTorch or download data."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def collect_report(workspace: Path) -> dict:
    """Collect host information without changing the system."""
    usage = shutil.disk_usage(workspace)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_logical_cores": os.cpu_count(),
        "disk_free_gib": round(usage.free / 1024**3, 2),
        "workspace": str(workspace.resolve()),
        "torch_installed": importlib.util.find_spec("torch") is not None,
    }
    try:
        import psutil
    except ImportError:
        report["ram_total_gib"] = None
        report["ram_available_gib"] = None
    else:
        memory = psutil.virtual_memory()
        report["ram_total_gib"] = round(memory.total / 1024**3, 2)
        report["ram_available_gib"] = round(memory.available / 1024**3, 2)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect CPU training prerequisites (no model load)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Path to check disk space")
    args = parser.parse_args(argv)
    if not args.workspace.is_dir():
        parser.error("--workspace must be an existing directory")
    report = collect_report(args.workspace)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for key, value in report.items():
            print(f"{key}: {value if value is not None else 'unavailable'}")
        print("Note: This is an inventory, not a training benchmark or GPU capability test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
