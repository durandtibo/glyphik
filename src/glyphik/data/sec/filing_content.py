r"""Define utilities for extracting content from an edgar Filing
object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from edgar import Filing

__all__ = ["extract_filing_content"]

FilingFormat = Literal["text", "markdown", "html", "xml", "full_text_submission"]


def extract_filing_content(filing: Filing, format_type: FilingFormat = "text") -> str | None:
    """Extract content from an edgartools Filing object in the specified
    format.

    Args:
        filing: The edgartools Filing object.
        format_type: The desired output format.

    Returns:
        str | None: The extracted content, or None if the format is not applicable
                    (e.g., 'xml' for non-XML filings).

    Raises:
        ValueError: If an unsupported format_type is provided.
    """
    valid_formats = get_args(FilingFormat)
    format_val = str(format_type).lower()
    if format_val not in valid_formats:
        msg = f"Invalid format '{format_type}'. Must be one of: {', '.join(valid_formats)}"
        raise ValueError(msg)

    try:
        extraction_method = getattr(filing, format_val)
    except AttributeError as exc:
        msg = f"The provided Filing object does not support the '{format_val}' method."
        raise ValueError(msg) from exc

    return extraction_method()
