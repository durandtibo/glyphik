r"""Contain utilities for optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_edgar",
    "edgar_available",
    "is_edgar_available",
    "raise_edgar_missing_error",
]

from glyphik.utils.imports.edgar import (
    check_edgar,
    edgar_available,
    is_edgar_available,
    raise_edgar_missing_error,
)
