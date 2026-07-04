"""Define a processor that redownloads a SEC filing if it cannot be
loaded from disk or fails an SGML validity check."""

from __future__ import annotations

__all__ = ["RedownloadSecFilingProcessor"]

import logging
from typing import Any

from coola.display import InlineDisplayMixin
from zenpyre.data_processors.base import BaseProcessor

from glyphik.data.sec import SecFilingRecord, has_valid_sgml
from glyphik.utils.imports import is_edgar_available

if is_edgar_available():  # pragma: no cover
    from edgar import Company

logger: logging.Logger = logging.getLogger(__name__)


class RedownloadSecFilingProcessor(
    BaseProcessor[SecFilingRecord, SecFilingRecord], InlineDisplayMixin
):
    """Processor that redownloads a filing when it is missing or invalid on disk.

        Attempts to load the filing referenced by a
        :class:`~glyphik.data.sec.SecFilingRecord` via
        :meth:`~glyphik.data.sec.SecFilingRecord.load_filing`. The filing
        is re-fetched from SEC EDGAR and saved back to the record's
        ``filepath`` if either of the following holds:

        - The filing cannot be loaded (e.g. the cached file is missing or
          corrupted).
        - ``check_sgml`` is ``True`` and the loaded filing fails
          :func:`~glyphik.data.sec.has_valid_sgml`.

        The redownload uses the record's ``cik``, ``form``, and
        ``accession_number`` metadata to locate the filing on EDGAR. The
        record itself is returned unchanged; this processor's effect is
        the side effect of restoring the file on disk.

        This processor is designed to be used element-wise inside a
        :class:`~zenpyre.data_processors.SequenceProcessor` to repair a
        list of :class:`~glyphik.data.sec.SecFilingRecord` instances before
        they are loaded further downstream.

    Args:
            check_sgml: If ``True``, also redownloads the filing when it
                loads successfully but fails the SGML validity check via
                :func:`~glyphik.data.sec.has_valid_sgml`. Defaults to
                ``False``, since this check requires loading and parsing
                the filing and is not always necessary.

    Example:
    ```pycon
    >>> from glyphik.data.sec import SecFilingRecord
    >>> from glyphik.data_processors import RedownloadSecFilingProcessor
    >>> processor = RedownloadSecFilingProcessor(check_sgml=True)
    >>> record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    >>> record = processor.process(record)  # doctest: +SKIP

    ```
    """

    def __init__(self, check_sgml: bool = False) -> None:
        self._check_sgml = check_sgml

    def process(self, data: SecFilingRecord) -> SecFilingRecord:
        """Redownload the filing to disk if it is missing or invalid.

        Calls :meth:`~glyphik.data.sec.SecFilingRecord.load_filing` to
        check whether the filing is available. If it cannot be loaded,
        or if :attr:`_check_sgml` is ``True`` and the loaded filing
        fails :func:`~glyphik.data.sec.has_valid_sgml`, looks up the
        filing on SEC EDGAR using the record's ``cik``, ``form``, and
        ``accession_number`` metadata, and saves it to the record's
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
        needs_redownload = filing is None or (self._check_sgml and not has_valid_sgml(filing))

        if needs_redownload:
            filepath = data.metadata["filepath"]
            company = Company(data.metadata["cik"])
            filings = company.get_filings(form=data.metadata["form"]).filter(
                accession_number=data.metadata["accession_number"]
            )
            if len(filings) == 1:
                filings[0].save(filepath)
            else:
                logger.warning(
                    "Could not uniquely identify filing to redownload for "
                    "cik=%s form=%s accession_number=%s (found %d matches)",
                    data.metadata["cik"],
                    data.metadata["form"],
                    data.metadata["accession_number"],
                    len(filings),
                )

        return data

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"check_sgml": self._check_sgml}
