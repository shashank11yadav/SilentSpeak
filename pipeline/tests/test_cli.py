from __future__ import annotations

import hashlib
from pathlib import Path

from typer.testing import CliRunner

from silentspeak.cli import app
from silentspeak.paths import CONFIG_DIR


def test_smoke_runs_the_whole_pipeline(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["smoke", "--config", str(CONFIG_DIR / "smoke.toml"), "--keep", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "smoke ok" in result.output
    assert (tmp_path / "models" / "model.onnx").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "replay" / "index.json").exists()


def test_help_lists_every_command() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "extract",
        "prepare",
        "train",
        "evaluate",
        "export",
        "export-fixtures",
        "replay-clips",
        "smoke",
        "verify",
    ):
        assert cmd in result.output


def test_verify_checks_every_complete_download(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "s1.zip").write_bytes(b"one")
    (raw / "alignments.zip").write_bytes(b"two")
    good = hashlib.md5(b"one").hexdigest()  # noqa: S324
    lines = [
        f"s1.zip\tmd5:{good}\t3\thttps://e/s1",
        f"alignments.zip\tmd5:{'0' * 32}\t3\thttps://e/a",
        f"s2.zip\tmd5:{'0' * 32}\t9\thttps://e/s2",  # not downloaded yet
    ]
    (tmp_path / "manifest.tsv").write_text("\n".join(lines) + "\n")
    result = CliRunner().invoke(app, ["verify", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "ok s1.zip" in result.output
    assert "alignments.zip" in result.output
    assert "s2.zip" not in result.output
