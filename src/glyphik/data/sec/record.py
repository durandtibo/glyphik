r"""Contain SEC records."""

from __future__ import annotations

__all__ = ["SecFilingRecord"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zenpyre.records import Record

from glyphik.utils.imports import is_edgar_available

if TYPE_CHECKING or is_edgar_available():
    from edgar import Filing
else:  # pragma: no cover
    from glyphik.utils.fallback.edgar import Filing


@dataclass(frozen=True)
class SecFilingRecord(Record):
    """A downloaded SEC filing record containing metadata and a
    reference to the filing saved on disk.

    The filing itself is not stored in memory — call
    :meth:`load_filing` to load it from disk on demand.  Use
    :meth:`from_metadata` as the preferred constructor to automatically
    derive a stable UUID from the metadata.

    Args:
        id: Unique identifier for the filing, typically a UUID derived
            from the metadata via :func:`~zenpyre.hashing.hash_dict_uuid`.
        metadata: Dict of metadata associated with the filing
            (e.g. accession number, CIK, ticker, form type, filepath).
    """

    def load_filing(self) -> Filing:
        """Load and return the filing from disk.

        Reads the file at ``metadata["filepath"]`` and returns the
        deserialised :class:`~edgar.Filing` object via
        :meth:`~edgar.Filing.load`.

        Returns:
            The :class:`~edgar.Filing` instance loaded from disk.

        Raises:
            ValueError: If ``"filepath"`` is not set in ``metadata``.
            FileNotFoundError: If the file at ``metadata["filepath"]``
                does not exist.
        """
        filepath = self.metadata.get("filepath")
        if filepath is None:
            msg = "Cannot load filing: 'filepath' is not set in metadata"
            raise ValueError(msg)
        return Filing.load(filepath)
