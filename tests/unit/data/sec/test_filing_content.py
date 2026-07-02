from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glyphik.data.sec.filing_content import FilingFormat, extract_filing_content
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


############################################
#     Tests for extract_filing_content     #
############################################


def test_extract_filing_content_default_format(mock_filing: Filing) -> None:
    """Test that the function defaults to 'text' if no format is
    provided."""
    result = extract_filing_content(mock_filing)
    assert result == "Sample text content"
    mock_filing.text.assert_called_once()


@pytest.mark.parametrize(
    ("format_type", "expected_result"),
    [
        ("text", "Sample text content"),
        ("markdown", "# Sample Markdown"),
        ("html", "<p>Sample HTML</p>"),
        ("xml", "<xml>Sample</xml>"),
        ("full_text_submission", "<SEC-DOCUMENT>Sample</SEC-DOCUMENT>"),
    ],
)
def test_extract_filing_content_valid_formats(
    mock_filing: Filing, format_type: FilingFormat, expected_result: str
) -> None:
    """Test that all valid Literal formats successfully extract data."""
    result = extract_filing_content(mock_filing, format_type)
    assert result == expected_result
    getattr(mock_filing, format_type).assert_called_once()


@pytest.mark.parametrize("format_type", ["TEXT", "Markdown", "hTmL"])
def test_extract_filing_content_case_insensitivity(
    mock_filing: Filing, format_type: FilingFormat
) -> None:
    """Test that uppercase or mixed-case format strings are handled
    safely."""
    extract_filing_content(mock_filing, format_type)
    getattr(mock_filing, format_type.lower()).assert_called_once()


def test_extract_filing_content_xml_returns_none(mock_filing: Filing) -> None:
    """Test the specific edge case where a non-XML filing returns
    None."""
    mock_filing.xml.return_value = None
    result = extract_filing_content(mock_filing, "xml")
    assert result is None
    mock_filing.xml.assert_called_once()


def test_extract_filing_content_invalid_format(mock_filing: Filing) -> None:
    """Test that passing an unsupported format raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid format 'pdf'"):
        extract_filing_content(mock_filing, "pdf")


def test_extract_filing_content_missing_attribute_raises_error() -> None:
    """Test that a ValueError is raised if the Filing object is missing
    the method."""
    bad_filing = BadFilingMock()
    with pytest.raises(ValueError, match="does not support the 'text' method"):
        extract_filing_content(bad_filing, "text")
