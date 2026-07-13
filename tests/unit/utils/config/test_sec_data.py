from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from glyphik.data.sec import CompanyIdentifier, SecForm
from glyphik.utils.config import SecDataConfig


def _make_config(**overrides: Any) -> SecDataConfig:
    """Return a SecDataConfig instance for testing."""
    kwargs = {
        "companies": None,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    }
    kwargs.update(overrides)
    return SecDataConfig(**kwargs)


###################################
#     Tests for SecDataConfig     #
###################################


# --- Date normalization ---


def test_sec_data_config_converts_start_date_from_str() -> None:
    config = _make_config(start_date="2023-01-01")
    assert config.start_date == date(2023, 1, 1)


def test_sec_data_config_converts_end_date_from_str() -> None:
    config = _make_config(end_date="2023-12-31")
    assert config.end_date == date(2023, 12, 31)


def test_sec_data_config_accepts_start_date_as_date() -> None:
    config = _make_config(start_date=date(2023, 1, 1))
    assert config.start_date == date(2023, 1, 1)


def test_sec_data_config_accepts_end_date_as_date() -> None:
    config = _make_config(end_date=date(2023, 12, 31))
    assert config.end_date == date(2023, 12, 31)


def test_sec_data_config_start_date_is_date_instance_after_init() -> None:
    config = _make_config(start_date="2023-01-01")
    assert isinstance(config.start_date, date)


def test_sec_data_config_end_date_is_date_instance_after_init() -> None:
    config = _make_config(end_date="2023-12-31")
    assert isinstance(config.end_date, date)


# --- Date range validation ---


def test_sec_data_config_accepts_start_date_equal_to_end_date() -> None:
    config = _make_config(start_date="2023-06-15", end_date="2023-06-15")
    assert config.start_date == config.end_date == date(2023, 6, 15)


def test_sec_data_config_accepts_start_date_before_end_date() -> None:
    config = _make_config(start_date="2023-01-01", end_date="2023-12-31")
    assert config.start_date < config.end_date


def test_sec_data_config_raises_when_start_date_after_end_date() -> None:
    with pytest.raises(ValueError, match=r"start_date .* must be <= end_date"):
        _make_config(start_date="2023-12-31", end_date="2023-01-01")


def test_sec_data_config_raises_when_start_date_after_end_date_mixed_types() -> None:
    with pytest.raises(ValueError, match=r"start_date .* must be <= end_date"):
        _make_config(start_date=date(2023, 12, 31), end_date="2023-01-01")


# --- forms normalization ---


def test_sec_data_config_default_forms() -> None:
    config = _make_config()
    assert config.forms == (SecForm.TEN_K, SecForm.TEN_Q)


def test_sec_data_config_converts_forms_list_to_tuple() -> None:
    config = _make_config(forms=[SecForm.TEN_K])
    assert config.forms == (SecForm.TEN_K,)
    assert isinstance(config.forms, tuple)


def test_sec_data_config_converts_forms_generator_to_tuple() -> None:
    config = _make_config(forms=(f for f in [SecForm.TEN_K, SecForm.TEN_Q]))
    assert config.forms == (SecForm.TEN_K, SecForm.TEN_Q)


# --- companies ---


def test_sec_data_config_accepts_none_companies() -> None:
    config = _make_config(companies=None)
    assert config.companies is None


def test_sec_data_config_accepts_companies_tuple() -> None:
    companies = (CompanyIdentifier(cik=320193, ticker="AAPL"),)
    config = _make_config(companies=companies)
    assert config.companies == companies


# --- Immutability ---


def test_sec_data_config_is_frozen() -> None:
    config = _make_config()
    with pytest.raises(AttributeError):
        config.start_date = date(2024, 1, 1)  # type: ignore[misc]


def test_sec_data_config_is_hashable() -> None:
    config = _make_config()
    assert hash(config) is not None
