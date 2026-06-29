from __future__ import annotations

import pytest

from glyphik.testing.fixtures import edgar_available, edgar_not_available
from glyphik.utils.imports import check_edgar, is_edgar_available

######################
#     edgar     #
######################


@edgar_available
def test_check_edgar_with_package() -> None:
    check_edgar()


@edgar_not_available
def test_check_edgar_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'edgar' package is required but not installed."):
        check_edgar()


@edgar_available
def test_is_edgar_available_true() -> None:
    assert is_edgar_available()


@edgar_not_available
def test_is_edgar_available_false() -> None:
    assert not is_edgar_available()
