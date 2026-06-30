from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from glyphik.data.sec import SecFilingRecord, SecForm, fetch_filings
from glyphik.testing.fixtures import edgar_available

if TYPE_CHECKING:
    from pathlib import Path

#####################################
#     Tests for fetch_filings       #
#####################################


@edgar_available
def test_fetch_filings(tmp_path: Path) -> None:
    docs = fetch_filings(
        cik=320193,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        output_dir=tmp_path,
        forms=[SecForm.TEN_K, SecForm.TEN_Q],
    )

    assert len(docs) == 4
    assert all(isinstance(doc, SecFilingRecord) for doc in docs)
    assert docs[0].metadata["form"] == SecForm.TEN_K
    assert docs[1].metadata["form"] == SecForm.TEN_Q
    assert docs[2].metadata["form"] == SecForm.TEN_Q
    assert docs[3].metadata["form"] == SecForm.TEN_Q
