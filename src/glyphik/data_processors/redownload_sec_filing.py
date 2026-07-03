"""Define a processor that redownloads a SEC filing if it cannot be
loaded from disk."""

from __future__ import annotations

__all__ = ["RedownloadSecFilingProcessor"]

import logging
from typing import Any

from coola.display import InlineDisplayMixin
from zenpyre.data_processors.base import BaseProcessor

from glyphik.data.sec import SecFilingRecord
from glyphik.utils.imports import is_edgar_available

if is_edgar_available():  # pragma: no cover
    from edgar import Company

logger: logging.Logger = logging.getLogger(__name__)


class RedownloadSecFilingProcessor(
    BaseProcessor[SecFilingRecord, SecFilingRecord], InlineDisplayMixin
):
    """Processor that redownloads a filing when it is missing on disk.

        Attempts to load the filing referenced by a
        :class:`~glyphik.data.sec.SecFilingRecord` via
        :meth:`~glyphik.data.sec.SecFilingRecord.load_filing`. If the
        filing cannot be loaded (e.g. the cached file is missing or
        corrupted), it is re-fetched from SEC EDGAR using the record's
        ``cik``, ``form``, and ``accession_no`` metadata, and saved back
        to the record's ``filepath``. The record itself is returned
        unchanged; this processor's effect is the side effect of restoring
        the file on disk.

        This processor is designed to be used element-wise inside a
        :class:`~zenpyre.data_processors.SequenceProcessor` to repair a
        list of :class:`~glyphik.data.sec.SecFilingRecord` instances before
        they are loaded further downstream.

    Example:
    ```pycon
    >>> from glyphik.data.sec import SecFilingRecord
    >>> from glyphik.data_processors import RedownloadSecFilingProcessor
    >>> processor = RedownloadSecFilingProcessor()
    >>> record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    >>> record = processor.process(record)  # doctest: +SKIP

    ```
    """

    def process(self, data: SecFilingRecord) -> SecFilingRecord:
        """Redownload the filing to disk if it cannot currently be
        loaded.

        Calls :meth:`~glyphik.data.sec.SecFilingRecord.load_filing` to
        check whether the filing is available. If it is not, looks up
        the filing on SEC EDGAR using the record's ``cik``, ``form``,
        and ``accession_no`` metadata, and saves it to the record's
        ``filepath`` so that subsequent loads succeed.

        Args:
            data: The :class:`~glyphik.data.sec.SecFilingRecord` to
                check and, if necessary, repair.

        Returns:
            The same :class:`~glyphik.data.sec.SecFilingRecord` that
            was passed in, unmodified. The filing is only redownloaded
            as a side effect on disk; the record's metadata and ``id``
            are not changed.
        """
        filing = data.load_filing()
        if filing is None:
            filepath = data.metadata["filepath"]
            company = Company(data.metadata["cik"])
            filings = company.get_filings(form=data.metadata["form"]).filter(
                accession_number=data.metadata["accession_no"]
            )
            if len(filings) == 1:
                filings[0].save(filepath)
            else:
                logger.warning(
                    "Could not uniquely identify filing to redownload for "
                    "cik=%s form=%s accession_no=%s (found %d matches)",
                    data.metadata["cik"],
                    data.metadata["form"],
                    data.metadata["accession_no"],
                    len(filings),
                )

        return data

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}
