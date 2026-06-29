from __future__ import annotations

from glyphik.data.sec import fetch_ticker_from_cik
from glyphik.testing.fixtures import edgar_available

##################################################
#   Tests for fetch_ticker_from_cik              #
##################################################

# --- return value ---


@edgar_available
def test_fetch_ticker_from_cik_returns_primary_ticker() -> None:
    assert fetch_ticker_from_cik(320193) == "AAPL"
