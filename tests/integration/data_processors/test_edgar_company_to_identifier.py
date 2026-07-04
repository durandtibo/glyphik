from __future__ import annotations

from glyphik.data.sec import CompanyIdentifier
from glyphik.data_processors import EdgarCompanyToIdentifierProcessor
from glyphik.testing.fixtures import edgar_available
from glyphik.utils.imports import is_edgar_available

if is_edgar_available():
    import edgar

######################################################
#   Tests for EdgarCompanyToIdentifierProcessor      #
######################################################


@edgar_available
def test_edgar_company_to_identifier_processor_process() -> None:
    company = edgar.Company("AAPL")
    result = EdgarCompanyToIdentifierProcessor().process(company)
    assert result == CompanyIdentifier(cik=320193, ticker="AAPL")
