r"""Contain utilities for optional edgartools dependency."""

from __future__ import annotations

__all__ = [
    "check_edgartools",
    "edgartools_available",
    "is_edgartools_available",
    "raise_edgartools_missing_error",
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


def check_edgartools() -> None:
    r"""Check if the ``edgartools`` package is installed.

    Raises:
        RuntimeError: if the ``edgartools`` package is not installed.

    Example:
        ```pycon
        >>> from glyphik.utils.imports import check_edgartools
        >>> check_edgartools()

        ```
    """
    if not is_edgartools_available():
        raise_edgartools_missing_error()


@lru_cache(1)
def is_edgartools_available() -> bool:
    r"""Indicate if the ``edgartools`` package is installed or not.

    Returns:
        ``True`` if ``edgartools`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from glyphik.utils.imports import is_edgartools_available
        >>> is_edgartools_available()

        ```
    """
    return package_available("edgartools")


def edgartools_available[F: "Callable[..., Any]"](fn: F) -> F:
    r"""Implement a decorator to execute a function only if
    ``edgartools`` package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``edgartools`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from glyphik.utils.imports import edgartools_available
        >>> @edgartools_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_edgartools_available)


def raise_edgartools_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``edgartools`` package is
    missing."""
    raise_package_missing_error("edgartools", "langchain-chroma")
