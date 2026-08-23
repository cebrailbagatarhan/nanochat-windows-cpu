from pathlib import Path

import pytest

from inference_cpu import resolve_checkpoint


def test_resolve_checkpoint_prefers_stronger_model(tmp_path: Path):
    (tmp_path / "best_model.pt").write_bytes(b"base")
    (tmp_path / "model_medium.pt").write_bytes(b"medium")
    (tmp_path / "model_max.pt").write_bytes(b"max")

    selected = resolve_checkpoint(None, tmp_path)

    assert selected == tmp_path / "model_max.pt"


def test_resolve_checkpoint_honors_explicit_path(tmp_path: Path):
    explicit = tmp_path / "custom.pt"
    explicit.write_bytes(b"checkpoint")

    assert resolve_checkpoint(str(explicit), tmp_path) == explicit


def test_resolve_checkpoint_reports_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(None, tmp_path)
