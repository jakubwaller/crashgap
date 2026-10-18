from __future__ import annotations

from pathlib import Path

from conftest import add_crash
from streamlit.testing.v1 import AppTest

from crashgap.db import connect

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_loaders_rerun_when_the_db_file_changes(tmp_path, monkeypatch):
    db = tmp_path / "crashgap.db"
    monkeypatch.setenv("CRASHGAP_DB", str(db))

    at = AppTest.from_file(APP).run()
    assert "No data yet" in at.warning[0].value

    # The cached "no data" answer must not outlive the DB changing under it.
    add_crash(connect(db), 1, [{"sex": 2}, {"sex": 1}])
    at = AppTest.from_file(APP).run()
    assert "No results yet" in at.warning[0].value
