r"""Contain code to explore yfinance lib."""

from __future__ import annotations

import logging

import yfinance as yf

from glyphik.utils.logging import log_pretty

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:  # noqa: D103
    dat = yf.Ticker("MSFT")
    log_pretty(dat.info, title="info")
    log_pretty(dat.calendar, title="calendar")
    log_pretty(dat.quarterly_income_stmt, title="quarterly_income_stmt")
    logger.info(type(dat.quarterly_income_stmt))
    log_pretty(dat.analyst_price_targets, title="analyst_price_targets")
    # logger.info(dat.history(period='1mo'))

    # get list of quotes
    quotes = yf.Search("AAPL", max_results=10).quotes
    log_pretty(quotes, title="quotes")

    # get list of news
    news = yf.Search("Google", news_count=10).news
    log_pretty(news, title="news")

    # get list of related research
    research = yf.Search("apple", include_research=True).research
    log_pretty(research, title="research")


if __name__ == "__main__":
    main()
