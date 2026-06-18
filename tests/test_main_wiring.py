"""Smoke test: importing app.main registers the data layer + resume callback."""

from __future__ import annotations


def test_main_registers_data_layer_and_resume() -> None:
    import app.main  # noqa: F401, I001  (import triggers the chainlit decorators)
    from chainlit.config import config

    assert config.code.data_layer is not None
    assert config.code.on_chat_resume is not None
    # The factory returns the SQLAlchemy data layer over our SQLite file.
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    layer = config.code.data_layer()
    assert isinstance(layer, SQLAlchemyDataLayer)
    assert layer._conninfo.startswith("sqlite+aiosqlite:///")
