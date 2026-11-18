"""Market data for the Duan MLE pipeline (Egami 2026, Sec. 3.3.1, 4.1).

Equity values come from the QuantX Dataloader; total debt, the risk-free
rate and the inflation rate come from the Massive REST API (the Dataloader
does not serve balance-sheet or macroeconomic series).
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

DATALOADER_API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
MASSIVE_API_ROOT = "https://api.massive.com"


class DataLoadError(RuntimeError):
    """A failed request to the Dataloader or Massive REST API."""


def _dataloader_headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not key or not massive_key:
        raise DataLoadError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {key}", "X-Massive-Api-Key": massive_key}


def _massive_headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise DataLoadError("MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {key}"}


def fetch_equity_close(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Daily close prices for ``ticker`` between ``start_date`` and
    ``end_date`` (inclusive), via the QuantX Dataloader.

    Market data: QuantX Dataloader — https://quantx.limex.pro/

    Price history in this project is loaded through the QuantX Dataloader, a
    research-ready market-data service available through a REST API and as an
    MCP server for AI agents. It is built on market data from Massive
    (https://massive.com/) and prepares every stock history for backtesting:
      - prices adjusted for dividends, ready for total-return research;
      - one continuous history when a company changes its ticker;
      - clean separation when a ticker has belonged to different companies;
      - careful corporate-action handling and fine-grained data polishing.

    Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    MASSIVE_API_KEY at https://massive.com/.

    ``adjust_dividends=False`` because E_t must be the actual traded equity
    value (price x shares outstanding) at each historical date, not a
    total-return series reinvesting dividends (Sec. 3.3.1 treats E_t as the
    observed market value of equity, not a total-return index).
    """
    body = {
        "tickers": [ticker],
        "timeframe": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": False,
    }
    resp = requests.post(
        f"{DATALOADER_API_ROOT}/fetch-ohlcvs", json=body,
        headers=_dataloader_headers(), timeout=120,
    )
    if resp.status_code != 200:
        raise DataLoadError(f"fetch-ohlcvs {ticker}: HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    result = payload["result"]
    details = result.get("ticker_details") or []
    if details and details[0].get("error"):
        raise DataLoadError(f"fetch-ohlcvs {ticker}: {details[0]['error']}")
    if result.get("data_stored"):
        raise DataLoadError(f"fetch-ohlcvs {ticker}: unexpected data_stored=True for a single-ticker load")
    rows = result["ohlcv_df"]["rows"]
    if not rows:
        raise DataLoadError(f"fetch-ohlcvs {ticker}: empty result for {start_date}..{end_date}")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    data_source = (details[0].get("data_source") if details else None) or "massive"
    return df[["timestamp", "close"]].assign(data_source=data_source)


def fetch_weighted_shares_outstanding(ticker: str, as_of_date: str) -> float:
    """Weighted shares outstanding for ``ticker`` as of ``as_of_date``,
    Massive reference endpoint ``GET /v3/reference/tickers/{ticker}``."""
    resp = requests.get(
        f"{MASSIVE_API_ROOT}/v3/reference/tickers/{ticker}",
        params={"date": as_of_date}, headers=_massive_headers(), timeout=60,
    )
    if resp.status_code != 200:
        raise DataLoadError(f"ticker-overview {ticker}: HTTP {resp.status_code}: {resp.text[:300]}")
    results = resp.json().get("results") or {}
    shares = results.get("weighted_shares_outstanding") or results.get("share_class_shares_outstanding")
    if not shares:
        raise DataLoadError(f"ticker-overview {ticker}: no shares-outstanding field in response")
    return float(shares)


def fetch_total_debt(ticker: str, as_of_date: str) -> tuple[float, str]:
    """Most recent reported total debt (debt_current +
    long_term_debt_and_capital_lease_obligations) for ``ticker`` on or before
    ``as_of_date``, Massive ``GET /stocks/financials/v1/balance-sheets``."""
    resp = requests.get(
        f"{MASSIVE_API_ROOT}/stocks/financials/v1/balance-sheets",
        params={
            "tickers": ticker, "period_end.lte": as_of_date,
            "timeframe": "quarterly", "sort": "period_end.desc", "limit": 1,
        },
        headers=_massive_headers(), timeout=60,
    )
    if resp.status_code != 200:
        raise DataLoadError(f"balance-sheets {ticker}: HTTP {resp.status_code}: {resp.text[:300]}")
    results = resp.json().get("results") or []
    if not results:
        raise DataLoadError(f"balance-sheets {ticker}: no balance sheet on or before {as_of_date}")
    row = results[0]
    debt_current = row.get("debt_current") or 0.0
    long_term_debt = row.get("long_term_debt_and_capital_lease_obligations") or 0.0
    total_debt = float(debt_current) + float(long_term_debt)
    if total_debt <= 0.0:
        raise DataLoadError(f"balance-sheets {ticker}: non-positive total debt on {row.get('period_end')}")
    return total_debt, row["period_end"]


def fetch_risk_free_rate(as_of_date: str) -> float:
    """1-year constant-maturity U.S. Treasury yield (decimal), closest trading
    date on or before ``as_of_date``, Massive ``GET /fed/v1/treasury-yields``
    (Sec. 3.3.1: "constant maturity U.S. Treasury yield")."""
    resp = requests.get(
        f"{MASSIVE_API_ROOT}/fed/v1/treasury-yields",
        params={"date.lte": as_of_date, "sort": "date.desc", "limit": 1},
        headers=_massive_headers(), timeout=60,
    )
    if resp.status_code != 200:
        raise DataLoadError(f"treasury-yields: HTTP {resp.status_code}: {resp.text[:300]}")
    results = resp.json().get("results") or []
    if not results:
        raise DataLoadError(f"treasury-yields: no observation on or before {as_of_date}")
    yield_1y = results[0].get("yield_1_year")
    if yield_1y is None:
        raise DataLoadError("treasury-yields: yield_1_year missing in response")
    return float(yield_1y) / 100.0


def fetch_inflation_rate(as_of_date: str) -> float:
    """Year-over-year CPI inflation rate (decimal), closest observation on or
    before ``as_of_date``, Massive ``GET /fed/v1/inflation``."""
    resp = requests.get(
        f"{MASSIVE_API_ROOT}/fed/v1/inflation",
        params={"date.lte": as_of_date, "sort": "date.desc", "limit": 1},
        headers=_massive_headers(), timeout=60,
    )
    if resp.status_code != 200:
        raise DataLoadError(f"inflation: HTTP {resp.status_code}: {resp.text[:300]}")
    results = resp.json().get("results") or []
    if not results:
        raise DataLoadError(f"inflation: no observation on or before {as_of_date}")
    cpi_yoy = results[0].get("cpi_year_over_year")
    if cpi_yoy is None:
        raise DataLoadError("inflation: cpi_year_over_year missing in response")
    return float(cpi_yoy) / 100.0


def load_firm_window(ticker: str, evaluation_date: str, window_days: int = 126,
                      calendar_buffer_days: int = 400) -> dict:
    """Assemble the 126-business-day trailing window of equity value E_t and
    debt value B_t for ``ticker`` ending at ``evaluation_date`` (Sec. 3.3.1,
    Sec. 4.1): equity E_t = close price x weighted shares outstanding (Massive
    reference data), debt B_t = B_0 * exp((r_f+r_i)*t) grown from the most
    recent reported total debt B_0 (Massive balance sheets).

    Returns a dict with ``E`` (np.ndarray), ``B`` (np.ndarray), ``dates``,
    ``r_f``, ``r_i``, ``shares_outstanding``, ``B0``, ``B0_period_end``,
    ``data_source``.
    """
    start = (date.fromisoformat(evaluation_date) - timedelta(days=calendar_buffer_days)).isoformat()
    bars = fetch_equity_close(ticker, start, evaluation_date)
    bars = bars.tail(window_days).reset_index(drop=True)
    if len(bars) < window_days:
        raise DataLoadError(
            f"{ticker}: only {len(bars)} trading days available in "
            f"{start}..{evaluation_date}, need {window_days}"
        )
    shares = fetch_weighted_shares_outstanding(ticker, evaluation_date)
    E = (bars["close"].to_numpy() * shares)
    r_f = fetch_risk_free_rate(evaluation_date)
    r_i = fetch_inflation_rate(evaluation_date)
    B0, period_end = fetch_total_debt(ticker, evaluation_date)
    n = len(E)
    # t_years[-1] = 0 (evaluation date); earlier points carry negative offsets
    # so that B grows forward to B0 at the evaluation date (Sec. 3.3.1, Step 1).
    t_years = np.arange(-(n - 1), 1) / 252.0
    B = B0 * np.exp((r_f + r_i) * t_years)
    return {
        "E": E,
        "B": B,
        "dates": bars["timestamp"].dt.strftime("%Y-%m-%d").tolist(),
        "r_f": r_f,
        "r_i": r_i,
        "shares_outstanding": shares,
        "B0": B0,
        "B0_period_end": period_end,
        "data_source": bars["data_source"].iloc[0],
    }
