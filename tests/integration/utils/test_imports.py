from __future__ import annotations

import pytest

from glyphik.testing.fixtures import edgartools_available, edgartools_not_available
from glyphik.utils.imports import check_edgartools, is_edgartools_available

######################
#     edgartools     #
######################


@edgartools_available
def test_check_edgartools_with_package() -> None:
    check_edgartools()


@edgartools_not_available
def test_check_edgartools_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'edgartools' package is required but not installed."):
        check_edgartools()


@edgartools_available
def test_is_edgartools_available_true() -> None:
    assert is_edgartools_available()


@edgartools_not_available
def test_is_edgartools_available_false() -> None:
    assert not is_edgartools_available()
