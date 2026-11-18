"""Shared pytest fixtures and marker registration.

Market-data tests are marked ``market_data`` and are skipped when
``DATALOADER_API_KEY`` is not present in the environment, so the default test
run never depends on a live API call.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test loads a reduced real-data slice through data/data_loader.py"
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def firm_window_slice(request: pytest.FixtureRequest) -> dict:
    """A reduced one-firm, ~40-business-day slice loaded through
    data/data_loader.py, cached across test runs via pytest's cache so
    repeated test sessions do not re-download it."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set: skipping market-data test")
    cache_dir = request.config.cache.mkdir("firm_window_slice")
    cache_file = cache_dir / "F_2021-01-13.json"
    import json

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    from data.data_loader import load_firm_window

    data = load_firm_window("F", "2021-01-13", window_days=30, calendar_buffer_days=200)
    serializable = {**data, "E": list(data["E"]), "B": list(data["B"])}
    cache_file.write_text(json.dumps(serializable))
    return serializable
