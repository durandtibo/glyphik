from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glyphik.data.sec import extract_filing_content, normalize_content_format
from glyphik.data.sec.filing_content import _VALID_FORMATS
from glyphik.testing.fixtures import edgar_available
from glyphik.utils.imports import is_edgar_available

if is_edgar_available():
    from edgar import Filing


@pytest.fixture
def mock_filing() -> Filing:
    """Creates a mock Filing object with predefined return values for
    all expected extraction methods."""
    filing = MagicMock(spec=Filing)
    filing.text.return_value = "Sample text content"
    filing.markdown.return_value = "# Sample Markdown"
    filing.html.return_value = "<p>Sample HTML</p>"
    filing.xml.return_value = "<xml>Sample</xml>"
    filing.full_text_submission.return_value = "<SEC-DOCUMENT>Sample</SEC-DOCUMENT>"
    return filing


class BadFilingMock: ...


######################################################
#     Tests for normalize_content_format             #
######################################################


@pytest.mark.parametrize("content_format", sorted(_VALID_FORMATS))
def test_normalize_content_format_valid_lowercase(content_format: str) -> None:
    assert normalize_content_format(content_format) == content_format


@pytest.mark.parametrize(
    ("content_format", "expected"),
    [
        ("TEXT", "text"),
        ("Text", "text"),
        ("MarkDown", "markdown"),
        ("HTML", "html"),
        ("XML", "xml"),
        ("Full_Text_Submission", "full_text_submission"),
    ],
)
def test_normalize_content_format_case_insensitive(content_format: str, expected: str) -> None:
    assert normalize_content_format(content_format) == expected


def test_normalize_content_format_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"Invalid format 'pdf'"):
        normalize_content_format("pdf")


def test_normalize_content_format_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"Invalid format ''"):
        normalize_content_format("")


@pytest.mark.parametrize("bad_value", [None, 123, ["text"], {"format": "text"}])
def test_normalize_content_format_non_string_raises_value_error(bad_value: object) -> None:
    with pytest.raises(TypeError, match=r"content_format must be a string"):
        normalize_content_format(bad_value)


def test_normalize_content_format_preserves_original_value_in_error_message() -> None:
    """The error should echo back exactly what the caller passed, not
    the normalized form, so users can spot typos/casing issues."""
    with pytest.raises(ValueError, match=r"Invalid format 'PDF'"):
        normalize_content_format("PDF")


############################################
#     Tests for extract_filing_content     #
############################################


@edgar_available
def test_extract_filing_content_default_format(mock_filing: Filing) -> None:
    """Test that the function defaults to 'text' if no format is
    provided."""
    result = extract_filing_content(mock_filing)
    assert result == "Sample text content"
    mock_filing.text.assert_called_once()


@edgar_available
@pytest.mark.parametrize(
    ("content_format", "expected_result"),
    [
        ("text", "Sample text content"),
        ("markdown", "# Sample Markdown"),
        ("html", "<p>Sample HTML</p>"),
        ("xml", "<xml>Sample</xml>"),
        ("full_text_submission", "<SEC-DOCUMENT>Sample</SEC-DOCUMENT>"),
    ],
)
def test_extract_filing_content_valid_formats(
    mock_filing: Filing, content_format: str, expected_result: str
) -> None:
    """Test that all valid Literal formats successfully extract data."""
    result = extract_filing_content(mock_filing, content_format)
    assert result == expected_result
    getattr(mock_filing, content_format).assert_called_once()


@edgar_available
@pytest.mark.parametrize("content_format", ["TEXT", "Markdown", "hTmL"])
def test_extract_filing_content_case_insensitivity(
    mock_filing: Filing, content_format: str
) -> None:
    """Test that uppercase or mixed-case format strings are handled
    safely."""
    extract_filing_content(mock_filing, content_format)
    getattr(mock_filing, content_format.lower()).assert_called_once()


@edgar_available
def test_extract_filing_content_xml_returns_none(mock_filing: Filing) -> None:
    """Test the specific edge case where a non-XML filing returns
    None."""
    mock_filing.xml.return_value = None
    result = extract_filing_content(mock_filing, "xml")
    assert result is None
    mock_filing.xml.assert_called_once()


@edgar_available
def test_extract_filing_content_invalid_format(mock_filing: Filing) -> None:
    """Test that passing an unsupported format raises a ValueError."""
    with pytest.raises(ValueError, match=r"Invalid format 'pdf'"):
        extract_filing_content(mock_filing, "pdf")


@edgar_available
def test_extract_filing_content_missing_attribute_raises_error() -> None:
    """Test that a ValueError is raised if the Filing object is missing
    the method."""
    bad_filing = BadFilingMock()
    with pytest.raises(ValueError, match=r"does not support the 'text' method"):
        extract_filing_content(bad_filing, "text")


@edgar_available
def test_extract_filing_content_case_insensitivity_returns_expected(
    mock_filing: Filing,
) -> None:
    """Uppercase/mixed-case formats should not just dispatch, but return
    the same content as their lowercase equivalent."""
    result = extract_filing_content(mock_filing, "TEXT")
    assert result == "Sample text content"


@edgar_available
@pytest.mark.parametrize("content_format", ["text", "markdown", "xml"])
def test_extract_filing_content_missing_attribute_reflects_requested_format(
    content_format: str,
) -> None:
    """The missing-method error should name the specific format that was
    requested, not just a hardcoded one."""
    bad_filing = BadFilingMock()
    with pytest.raises(ValueError, match=f"does not support the '{content_format}' method"):
        extract_filing_content(bad_filing, content_format)


@edgar_available
def test_extract_filing_content_non_callable_attribute_raises_error(
    mock_filing: Filing,
) -> None:
    """A ValueError should be raised if the matched attribute exists but
    is not callable."""
    mock_filing.text = "not callable"
    with pytest.raises(ValueError, match=r"not callable"):
        extract_filing_content(mock_filing, "text")
