r"""Contain fallback implementations used when ``edgar`` dependency is
not available."""

from __future__ import annotations

__all__ = ["Company", "CompanyNotFoundError", "Filing", "edgar"]

from types import ModuleType
from typing import Any

from glyphik.utils.imports import raise_edgar_missing_error


class FakeClass:
    r"""Fake class that raises an error because edgar is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: edgar is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_edgar_missing_error()


Company = FakeClass
CompanyNotFoundError = FakeClass
Filing = FakeClass

entity: ModuleType = ModuleType("edgar.entity")
entity.CompanyNotFoundError = CompanyNotFoundError

# Create a fake edgar package
edgar: ModuleType = ModuleType("edgar")
edgar.entity = entity
edgar.Company = Company
edgar.Filing = Filing
