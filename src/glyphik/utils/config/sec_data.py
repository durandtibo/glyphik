r"""Contain a data ingestion config."""

from __future__ import annotations

__all__ = ["SecDataConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zenpyre.utils.config import ExtraFieldsConfig

from glyphik.data.sec import CompanyIdentifier, SecForm
from glyphik.utils.dates import coerce_to_date

if TYPE_CHECKING:
    from datetime import date


@dataclass(frozen=True)
class SecDataConfig(ExtraFieldsConfig):
    r"""Hold the configuration for a data ingestion pipelines.

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
    start_date: date
    end_date: date
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
        object.__setattr__(self, "start_date", coerce_to_date(self.start_date))
        object.__setattr__(self, "end_date", coerce_to_date(self.end_date))
        object.__setattr__(self, "forms", tuple(self.forms))
        if self.start_date > self.end_date:
            msg = f"start_date ({self.start_date}) must be <= end_date ({self.end_date})"
            raise ValueError(msg)

    def __hash__(self) -> int:
        # @dataclass(frozen=True) auto-generates a fresh __hash__ for
        # every dataclass-decorated class unless __hash__ is already
        # present in that class's own body — merely inheriting one
        # does not suppress the override. Delegating here (rather than
        # repeating its logic) keeps ExtraFieldsConfig.__hash__ the
        # single authoritative implementation.
        return ExtraFieldsConfig.__hash__(self)
