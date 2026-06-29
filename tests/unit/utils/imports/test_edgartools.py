from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from glyphik.utils.imports import (
    check_edgartools,
    edgartools_available,
    is_edgartools_available,
    raise_edgartools_missing_error,
)

logger = logging.getLogger(__name__)


MODULE = "glyphik.utils.imports.edgartools"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_edgartools_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


############################
#     edgartools     #
############################


def test_check_edgartools_with_package() -> None:
    with patch(f"{MODULE}.is_edgartools_available", lambda: True):
        check_edgartools()


def test_check_edgartools_without_package() -> None:
    with (
        patch(f"{MODULE}.is_edgartools_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'edgartools' package is required but not installed."),
    ):
        check_edgartools()


def test_is_edgartools_available() -> None:
    assert isinstance(is_edgartools_available(), bool)


def test_edgartools_available_with_package() -> None:
    with patch(f"{MODULE}.is_edgartools_available", lambda: True):
        fn = edgartools_available(my_function)
        assert fn(2) == 44


def test_edgartools_available_without_package() -> None:
    with patch(f"{MODULE}.is_edgartools_available", lambda: False):
        fn = edgartools_available(my_function)
        assert fn(2) is None


def test_edgartools_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_edgartools_available", lambda: True):

        @edgartools_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_edgartools_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_edgartools_available", lambda: False):

        @edgartools_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_edgartools_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'edgartools' package is required but not installed."):
        raise_edgartools_missing_error()
