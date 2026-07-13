r"""Define the base class to implement a pipelines."""

from __future__ import annotations

__all__ = ["BasePipeline"]

from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T")


class BasePipeline[T](ABC):
    """Abstract base class for pipelines.

    A pipelines encapsulates a sequence of steps that together produce a
    typed output. Subclasses must implement :meth:`run`, which runs
    the pipelines and returns its result.

    The generic parameter ``T`` defines the type of the pipelines's output,
    allowing static type checkers to verify that callers handle the result
    correctly.

    Example:
        ```pycon
        >>> from glyphik.pipelines import BasePipeline
        >>> class SumPipeline(BasePipeline[int]):
        ...     def __init__(self, values: list[int]) -> None:
        ...         self._values = values
        ...     def run(self) -> int:
        ...         return sum(self._values)
        ...
        >>> SumPipeline([1, 2, 3]).run()
        6

        ```
    """

    @abstractmethod
    def run(self) -> T:
        """Run the pipelines and return its result.

        Returns:
            The output produced by the pipelines. The type is determined
            by the generic parameter ``T`` of the concrete subclass.
        """
