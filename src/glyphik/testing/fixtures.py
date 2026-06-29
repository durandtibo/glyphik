r"""Define some pytest fixtures for testing.

`pytest` is required to use these fixtures.
"""

from __future__ import annotations

__all__ = ["edgartools_available", "edgartools_not_available"]

import pytest

from glyphik.utils.imports import is_edgartools_available

edgartools_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_edgartools_available(), reason="Requires edgartools"
)
edgartools_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_edgartools_available(), reason="Skip if edgartools is available"
)
