"""Define utilities for extracting content from an edgar Filing
object."""

from __future__ import annotations

__all__ = ["ContentFormat", "extract_filing_content", "normalize_content_format"]

from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from edgar import Filing


ContentFormat = Literal["text", "markdown", "html", "xml", "full_text_submission"]

_VALID_FORMATS = frozenset(get_args(ContentFormat))


def normalize_content_format(content_format: str) -> str:
    """Validate and normalize a content format string.

    Args:
        content_format: The format string to validate, e.g. ``"text"``
            or ``"TEXT"``. Matched case-insensitively against
            :data:`_VALID_FORMATS`.

    Returns:
        The lowercased, validated format string.

    Raises:
        ValueError: If ``content_format`` is not a string, or is not
            one of :data:`_VALID_FORMATS` (case-insensitively).
    """
    if not isinstance(content_format, str):
        msg = f"content_format must be a string, got {type(content_format).__name__}"
        raise TypeError(msg)

    format_val = content_format.lower()
    if format_val not in _VALID_FORMATS:
        msg = (
            f"Invalid format '{content_format}'. "
            f"Must be one of: {', '.join(sorted(_VALID_FORMATS))}"
        )
        raise ValueError(msg)

    return format_val


def extract_filing_content(filing: Filing, content_format: str = "text") -> str | None:
    """Extract content from an edgartools Filing object in the specified
    format.

    Args:
        filing: The edgartools Filing object.
        content_format: The desired output format. Matched case-insensitively
            against :data:`ContentFormat` (e.g. ``"TEXT"`` and ``"text"`` are
            equivalent).

    Returns:
        The extracted content. May be ``None`` if the
            underlying edgartools method itself returns ``None`` for this
            filing (e.g. calling ``.xml()`` on a filing with no XML
            representation) -- this function does not decide applicability
            itself, it simply forwards the result.

    Raises:
        ValueError: If ``content_format`` is not one of :data:`ContentFormat`,
            or if the given ``filing`` object does not implement the
            corresponding extraction method.
    """
    content_format = normalize_content_format(content_format)

    try:
        extraction_method = getattr(filing, content_format)
    except AttributeError as exc:
        msg = f"The provided Filing object does not support the '{content_format}' method."
        raise ValueError(msg) from exc

    try:
        return extraction_method()
    except TypeError as exc:
        msg = f"The '{content_format}' attribute on the provided Filing object is not callable."
        raise ValueError(msg) from exc
