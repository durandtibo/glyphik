from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from glyphik.utils.imports import (
    check_edgar,
    edgar_available,
    is_edgar_available,
    raise_edgar_missing_error,
)

logger = logging.getLogger(__name__)


MODULE = "glyphik.utils.imports.edgar"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_edgar_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#################
#     edgar     #
#################


def test_check_edgar_with_package() -> None:
    with patch(f"{MODULE}.is_edgar_available", lambda: True):
        check_edgar()


def test_check_edgar_without_package() -> None:
    with (
        patch(f"{MODULE}.is_edgar_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."),
    ):
        check_edgar()


def test_is_edgar_available() -> None:
    assert isinstance(is_edgar_available(), bool)


def test_edgar_available_with_package() -> None:
    with patch(f"{MODULE}.is_edgar_available", lambda: True):
        fn = edgar_available(my_function)
        assert fn(2) == 44


def test_edgar_available_without_package() -> None:
    with patch(f"{MODULE}.is_edgar_available", lambda: False):
        fn = edgar_available(my_function)
        assert fn(2) is None


def test_edgar_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_edgar_available", lambda: True):

        @edgar_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_edgar_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_edgar_available", lambda: False):

        @edgar_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_edgar_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        raise_edgar_missing_error()
