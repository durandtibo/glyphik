r"""Provide a utility to fetch the current S&P 1500 constituents from
Wikipedia."""

from __future__ import annotations

__all__ = ["Company", "fetch_companies", "load_or_fetch_companies"]

import logging
from typing import TYPE_CHECKING

from coola.utils.imports import is_pandas_available
from coola.utils.path import sanitize_path
from zenpyre.utils.dataclass_io import load_dataclasses, save_dataclasses

from glyphik.data.sp1500.cik import fill_missing_ciks
from glyphik.data.sp1500.company import Company

if TYPE_CHECKING:
    from pathlib import Path

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

_WIKIPEDIA_URLS: dict[str, str] = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P MidCap 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P SmallCap 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

_TICKER_COLUMNS: tuple[str, ...] = ("Symbol", "Ticker symbol", "Ticker")
_SECURITY_COLUMNS: tuple[str, ...] = ("Security", "Company")
_SECTOR_COLUMNS: tuple[str, ...] = ("GICS Sector",)
_SUB_INDUSTRY_COLUMNS: tuple[str, ...] = ("GICS Sub-Industry", "GICS Sub Industry")
_CIK_COLUMNS: tuple[str, ...] = ("CIK",)

_HEADERS: dict[str, str] = {"User-Agent": "Mozilla/5.0"}


def load_or_fetch_companies(
    path: Path | str | None, find_missing_ciks: bool = True
) -> list[Company]:
    """Load S&P 1500 companies from a cached JSON file, or fetch and
    cache them if no cache exists.

    If ``path`` is given and already exists, the companies are loaded
    from it via :func:`load_dataclasses`. Otherwise, the companies are
    fetched from Wikipedia via :func:`fetch_sp1500_companies`,
    optionally enriched with missing CIK numbers via
    :func:`fill_missing_ciks`, and -- if ``path`` is given -- saved to
    ``path`` via :func:`save_dataclasses` for future calls.

    Args:
        path: The path to the JSON cache file. If ``None``, caching is
            disabled entirely: companies are always freshly fetched and
            nothing is loaded from or saved to disk.
        find_missing_ciks: If ``True`` (the default), calls
            :func:`fill_missing_ciks` after fetching to look up CIK
            numbers for companies whose ``cik`` is ``None`` (e.g.
            S&P MidCap 400 companies, which Wikipedia does not list
            CIKs for).  Set to ``False`` to skip this step and save
            companies as-is.  Has no effect when loading from an
            existing cache.

    Returns:
        A list of :class:`~glyphik.data.sp1500.Company` instances,
        either loaded from the cache or freshly fetched and cached.

    Example:
        ```pycon
        >>> from glyphik.data.sp1500 import load_or_fetch_companies
        >>> companies = load_or_fetch_companies("sp1500.json")  # doctest: +SKIP

        ```
    """
    if path is not None:
        path = sanitize_path(path)
        if path.is_file():
            logger.info("Loading cached S&P 1500 companies from %s...", path)
            return load_dataclasses(path, Company)
        logger.info("No cache found at %s, fetching from Wikipedia...", path)
    else:
        logger.info("No cache path given, fetching from Wikipedia...")

    companies = fetch_companies()
    if find_missing_ciks:
        companies = fill_missing_ciks(companies)
    if path is not None:
        save_dataclasses(companies, path)
    return companies


def fetch_companies() -> list[Company]:
    """Fetch the current approximate S&P 1500 constituents from Wikipedia.

    Scrapes the constituent tables of the S&P 500, S&P MidCap 400, and
    S&P SmallCap 600 Wikipedia pages and returns one
    :class:`Company` per row across all three indices.  Each page
    is fetched and parsed independently via :func:`_find_constituent_table`
    and :func:`_parse_table`, then the results are concatenated.

    .. warning::
        This is **not** an authoritative or point-in-time accurate
        source.  Wikipedia is community-maintained and may lag behind
        official S&P Dow Jones Indices changes by days or weeks.  Use
        this for exploratory work only — not for production backtesting
        or anything requiring survivorship-bias-free, licensed index
        data.

    Returns:
        A list of :class:`Company` instances, one per
        constituent, across all three sub-indices.  Typically close to,
        but not exactly, 1500 entries (multiple share classes, recent
        changes, and scraping quirks can shift the count slightly).

    Raises:
        ValueError: If a constituent table, or a required column within
            it, cannot be found on one of the Wikipedia pages (e.g. the
            page structure changed).

    Example:
        ```python
        from zenpyre.data.sp1500 import fetch_sp1500_companies

        companies = fetch_sp1500_companies()  # doctest: +SKIP
        print(len(companies))  # doctest: +SKIP
        print(companies[0].ticker, companies[0].gics_sector)  # doctest: +SKIP
        ```
    """
    companies: list[Company] = []

    for index_name, url in _WIKIPEDIA_URLS.items():
        logger.info("Fetching %s constituents from %s...", index_name, url)
        tables = pd.read_html(url, storage_options=_HEADERS)
        table = _find_constituent_table(tables, index_name)
        index_companies = _parse_table(table, index_name)
        companies.extend(index_companies)
        logger.info("Found %s companies for %s", f"{len(index_companies):,}", index_name)

    logger.info("Total companies across S&P 1500: %s", f"{len(companies):,}")
    return companies


def _find_constituent_table(tables: list[pd.DataFrame], index_name: str) -> pd.DataFrame:
    """Return the constituent table among all tables parsed from a page.

    Wikipedia pages can contain several HTML tables (e.g. a sector
    summary table alongside the full constituent list).  This selects
    the first table that contains a recognised ticker column, as
    defined by :data:`_TICKER_COLUMNS`.

    Args:
        tables: The list of :class:`~pandas.DataFrame` instances
            returned by :func:`pandas.read_html` for a single page.
        index_name: The name of the S&P sub-index the page corresponds
            to (e.g. ``"S&P 500"``), used only for error reporting.

    Returns:
        The first :class:`~pandas.DataFrame` in ``tables`` that
        contains a recognised ticker column.

    Raises:
        ValueError: If none of the tables in ``tables`` contain a
            recognised ticker column.
    """
    for table in tables:
        if any(col in table.columns for col in _TICKER_COLUMNS):
            return table
    msg = f"Could not find a constituent table with a ticker column for {index_name}"
    raise ValueError(msg)


def _find_optional_column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name from ``candidates``, if
    any.

    Unlike :func:`_find_column`, this does not raise when no candidate
    is found — it is used for fields that may legitimately be absent
    from a given page (e.g. the CIK column on the S&P MidCap 400 page).

    Args:
        table: The :class:`~pandas.DataFrame` to search for a matching
            column.
        candidates: The candidate column names to look for, in order
            of preference.

    Returns:
        The first column name in ``candidates`` found in
        ``table.columns``, or ``None`` if none of them are present.
    """
    for candidate in candidates:
        if candidate in table.columns:
            return candidate
    return None


def _find_column(
    table: pd.DataFrame,
    candidates: tuple[str, ...],
    field_name: str,
    index_name: str,
) -> str:
    """Return the first matching, required column name from
    ``candidates``.

    Args:
        table: The :class:`~pandas.DataFrame` to search for a matching
            column.
        candidates: The candidate column names to look for, in order
            of preference.
        field_name: A human-readable name for the field being looked
            up (e.g. ``"ticker"``), used only for error reporting.
        index_name: The name of the S&P sub-index the table
            corresponds to (e.g. ``"S&P 500"``), used only for error
            reporting.

    Returns:
        The first column name in ``candidates`` found in
        ``table.columns``.

    Raises:
        ValueError: If none of ``candidates`` are present in
            ``table.columns``.
    """
    column = _find_optional_column(table, candidates)
    if column is None:
        msg = f"Could not find a {field_name!r} column for {index_name}"
        raise ValueError(msg)
    return column


def _parse_table(table: pd.DataFrame, index_name: str) -> list[Company]:
    """Convert a raw Wikipedia constituent table into a list of
    companies.

    Resolves the ticker, security, GICS sector, GICS sub-industry, and
    (optional) CIK columns via :func:`_find_column` and
    :func:`_find_optional_column`, then builds one :class:`Company`
    per row.

    Args:
        table: The raw constituent :class:`~pandas.DataFrame` for a
            single S&P sub-index, as returned by
            :func:`_find_constituent_table`.
        index_name: The name of the S&P sub-index the table
            corresponds to (e.g. ``"S&P 500"``), stored on each
            resulting :class:`Company` and used for error
            reporting.

    Returns:
        A list of :class:`Company` instances, one per row in
        ``table``.

    Raises:
        ValueError: If a required column (ticker, security, GICS
            sector, or GICS sub-industry) cannot be found in ``table``.
    """
    ticker_col = _find_column(table, _TICKER_COLUMNS, "ticker", index_name)
    security_col = _find_column(table, _SECURITY_COLUMNS, "security", index_name)
    sector_col = _find_column(table, _SECTOR_COLUMNS, "GICS Sector", index_name)
    sub_industry_col = _find_column(table, _SUB_INDUSTRY_COLUMNS, "GICS Sub-Industry", index_name)
    cik_col = _find_optional_column(table, _CIK_COLUMNS)

    return [
        Company(
            ticker=str(row[ticker_col]).strip(),
            cik=int(row[cik_col]) if cik_col is not None and pd.notna(row[cik_col]) else None,
            security=str(row[security_col]).strip(),
            gics_sector=str(row[sector_col]).strip(),
            gics_sub_industry=str(row[sub_industry_col]).strip(),
            index=index_name,
        )
        for _, row in table.iterrows()
    ]
