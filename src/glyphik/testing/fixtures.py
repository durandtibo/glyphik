r"""Define some pytest fixtures for testing.

`pytest` is required to use these fixtures.
"""

from __future__ import annotations

__all__ = ["edgar_available", "edgar_not_available"]

import pytest

from glyphik.utils.imports import is_edgar_available

edgar_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_edgar_available(), reason="Requires edgar"
)
edgar_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_edgar_available(), reason="Skip if edgar is available"
)
