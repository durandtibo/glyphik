"""Unit tests for the edgar fallback module."""

from __future__ import annotations

from types import ModuleType

import pytest

from glyphik.utils.fallback.edgar import Company, CompanyNotFoundError, edgar

#########################################
#     Tests for the edgar fallback      #
#########################################


# --- edgar module ---


def test_edgar_is_module_type() -> None:
    assert isinstance(edgar, ModuleType)


def test_edgar_module_name() -> None:
    assert edgar.__name__ == "edgar"


def test_edgar_has_company() -> None:
    assert hasattr(edgar, "Company")


def test_edgar_has_entity() -> None:
    assert hasattr(edgar, "entity")


def test_edgar_entity_is_module_type() -> None:
    assert isinstance(edgar.entity, ModuleType)


def test_edgar_entity_module_name() -> None:
    assert edgar.entity.__name__ == "edgar.entity"


def test_edgar_entity_has_company_not_found_error() -> None:
    assert hasattr(edgar.entity, "CompanyNotFoundError")


# --- Company ---


def test_company_is_class() -> None:
    assert isinstance(Company, type)


def test_company_instantiation_raises() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        Company()


def test_company_instantiation_with_args_raises() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        Company("AAPL")


def test_company_instantiation_with_kwargs_raises() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        Company(ticker="AAPL")


def test_edgar_company_instantiation_raises() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        edgar.Company()


# --- CompanyNotFoundError ---


def test_company_not_found_error_is_class() -> None:
    assert isinstance(CompanyNotFoundError, type)


def test_company_not_found_error_instantiation_raises() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        CompanyNotFoundError()


def test_edgar_entity_company_not_found_error_instantiation_raises() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        edgar.entity.CompanyNotFoundError()
