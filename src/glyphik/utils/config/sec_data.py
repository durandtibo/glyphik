r"""Contain a data ingestion config."""

from __future__ import annotations

__all__ = ["SecDataConfig"]

from dataclasses import dataclass
from datetime import date

from zenpyre.utils.config import ExtraFieldsConfig

from glyphik.data.sec import CompanyIdentifier, SecForm


@dataclass(frozen=True)
class SecDataConfig(ExtraFieldsConfig):
    r"""Hold the configuration for a data ingestion pipeline.

    Attributes:
        companies: The companies to ingest SEC filings for. ``None``
            means all companies (implementation-specific).
        start_date: The earliest date (inclusive) to ingest. Accepts
            a :class:`~datetime.date` or an ISO-formatted
            :class:`str`, which is automatically converted to a
            :class:`~datetime.date`.
        end_date: The latest date (inclusive) to ingest. Accepts a
            :class:`~datetime.date` or an ISO-formatted :class:`str`,
            which is automatically converted to a
            :class:`~datetime.date`.
        forms: The SEC forms to ingest. Accepts any iterable (e.g. a
            list) at construction time; always normalized to a
            ``tuple`` so the config stays hashable and immutable.

    Raises:
        ValueError: If ``start_date`` is after ``end_date``.
    """

    companies: tuple[CompanyIdentifier, ...] | None
    start_date: date | str
    end_date: date | str
    forms: tuple[str, ...] = (SecForm.TEN_K, SecForm.TEN_Q)

    def __post_init__(self) -> None:
        """Normalize dates and forms, and validate the date range.

        Converts ``start_date`` and ``end_date`` to
        :class:`~datetime.date` if given as ISO-formatted strings,
        converts ``forms`` to a tuple, and checks that ``start_date``
        does not come after ``end_date``.

        Raises:
            ValueError: If ``start_date`` is after ``end_date``.
        """
        start_date = (
            self.start_date
            if isinstance(self.start_date, date)
            else date.fromisoformat(self.start_date)
        )
        end_date = (
            self.end_date if isinstance(self.end_date, date) else date.fromisoformat(self.end_date)
        )
        object.__setattr__(self, "start_date", start_date)
        object.__setattr__(self, "end_date", end_date)
        object.__setattr__(self, "forms", tuple(self.forms))
        if start_date > end_date:
            msg = f"start_date ({start_date}) must be <= end_date ({end_date})"
            raise ValueError(msg)
