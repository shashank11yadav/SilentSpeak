from __future__ import annotations

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
    ):
        assert cmd in result.output
