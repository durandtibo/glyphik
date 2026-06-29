r"""Contain utilities for optional edgar dependency."""

from __future__ import annotations

__all__ = [
    "check_edgar",
    "edgar_available",
    "is_edgar_available",
    "raise_edgar_missing_error",
]

from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from coola.utils.imports import (
    decorator_package_available,
    package_available,
    raise_package_missing_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def check_edgar() -> None:
    r"""Check if the ``edgar`` package is installed.

    Raises:
        RuntimeError: if the ``edgar`` package is not installed.

    Example:
        ```pycon
        >>> from glyphik.utils.imports import check_edgar
        >>> check_edgar()

        ```
    """
    if not is_edgar_available():
        raise_edgar_missing_error()


@lru_cache(1)
def is_edgar_available() -> bool:
    r"""Indicate if the ``edgar`` package is installed or not.

    Returns:
        ``True`` if ``edgar`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from glyphik.utils.imports import is_edgar_available
        >>> is_edgar_available()

        ```
    """
    return package_available("edgar")


def edgar_available[F: "Callable[..., Any]"](fn: F) -> F:
    r"""Implement a decorator to execute a function only if
    ``edgar`` package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``edgar`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from glyphik.utils.imports import edgar_available
        >>> @edgar_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_edgar_available)


def raise_edgar_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``edgar`` package is
    missing."""
    raise_package_missing_error("edgar", "edgar")
