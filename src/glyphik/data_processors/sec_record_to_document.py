r"""Define a processor that converts a SecFilingRecord to a LangChain
Document."""

from __future__ import annotations

__all__ = ["SecFilingRecordToDocumentProcessor"]

import logging
from typing import Any

from coola.display import InlineDisplayMixin
from langchain_core.documents import Document
from zenpyre.data_processors.base import BaseProcessor

from glyphik.data.sec import SecFilingRecord
from glyphik.data.sec.filing_content import ContentFormat, extract_filing_content

logger: logging.Logger = logging.getLogger(__name__)


class SecFilingRecordToDocumentProcessor(
    BaseProcessor[SecFilingRecord, Document], InlineDisplayMixin
):
    """Processor that converts a :class:`~glyphik.data.sec.SecFilingRecord`
    to a :class:`~langchain_core.documents.Document`.

    Loads the filing from disk via
    :meth:`~glyphik.data.sec.SecFilingRecord.load_filing`, extracts its
    content in the specified format, and wraps it in a
    :class:`~langchain_core.documents.Document` with the record's
    metadata and ``id``.

    This processor is designed to be used element-wise inside a
    :class:`~zenpyre.data_processors.SequenceProcessor` to convert a
    list of :class:`~glyphik.data.sec.SecFilingRecord` instances into
    LangChain documents ready for indexing.

    Args:
        content_format: The content format to extract from the filing. One of:

            - ``"text"`` (default): clean plain text with HTML
              stripped — best for keyword search and simple NLP.
            - ``"markdown"``: HTML converted to Markdown — best for
              LLMs and RAG pipelines since it preserves table
              structure and formatting.
            - ``"html"``: raw HTML — best when downstream processing
              requires the original markup.

    Example:
        ```pycon
        >>> from glyphik.data.sec import SecFilingRecord
        >>> from glyphik.data_processors import SecFilingRecordToDocumentProcessor
        >>> processor = SecFilingRecordToDocumentProcessor()
        >>> record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
        >>> doc = processor.process(record)  # doctest: +SKIP

        ```
    """

    def __init__(self, content_format: ContentFormat = "text") -> None:
        self._content_format = content_format

    def process(self, data: SecFilingRecord) -> Document:
        """Load the filing from disk and return it as a Document.

        Calls :meth:`~glyphik.data.sec.SecFilingRecord.load_filing` to
        retrieve the filing, then extracts its content in the configured
        format and wraps it in a
        :class:`~langchain_core.documents.Document`.

        Args:
            data: The :class:`~glyphik.data.sec.SecFilingRecord` to
                convert.

        Returns:
            A :class:`~langchain_core.documents.Document` whose
            ``page_content`` is the filing content in :attr:`_content_format`,
            ``metadata`` is copied from the record, and ``id`` matches
            the record's ``id``.

        Raises:
            ValueError: If ``"filepath"`` is not set in the record's
                metadata.
            FileNotFoundError: If the filing file does not exist on
                disk.
        """
        filing = data.load_filing()

        page_content = extract_filing_content(filing, content_format=self._content_format)

        if page_content is None:
            # Fixed the string formatting for the ValueError
            msg = f"Filing content could not be extracted for record: {data.id}"
            raise ValueError(msg)

        logger.debug("Successfully processed filing record: %s", data.id)

        return Document(
            id=data.id,
            page_content=page_content,
            metadata=data.metadata.copy(),  # Copied to prevent LangChain from mutating the original
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"content_format": self._content_format}
