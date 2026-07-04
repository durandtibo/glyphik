from __future__ import annotations

from glyphik.data.sec import CompanyIdentifier
from glyphik.testing.fixtures import edgar_available
from glyphik.utils.imports import is_edgar_available

if is_edgar_available():
    import edgar


#######################################
#     Tests for CompanyIdentifier     #
#######################################


@edgar_available
def test_company_identifier_from_edgar_company_real_object() -> None:
    company = edgar.Company("AAPL")
    identifier = CompanyIdentifier.from_edgar_company(company)
    assert identifier == CompanyIdentifier(cik=320193, ticker="AAPL")
