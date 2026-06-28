r"""Define the base class to implement a pipeline."""

from __future__ import annotations

__all__ = ["BasePipeline"]

from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T")


class BasePipeline[T](ABC):
    """Abstract base class for pipelines.

    A pipeline encapsulates a sequence of steps that together produce a
    typed output. Subclasses must implement :meth:`execute`, which runs
    the pipeline and returns its result.

    The generic parameter ``T`` defines the type of the pipeline's output,
    allowing static type checkers to verify that callers handle the result
    correctly.

    Example:
        ```pycon
        >>> from glyphik.pipeline import BasePipeline
        >>> class SumPipeline(BasePipeline[int]):
        ...     def __init__(self, values: list[int]) -> None:
        ...         self._values = values
        ...     def execute(self) -> int:
        ...         return sum(self._values)
        ...
        >>> SumPipeline([1, 2, 3]).execute()
        6

        ```
    """

    @abstractmethod
    def execute(self) -> T:
        """Execute the pipeline and return its result.

        Returns:
            The output produced by the pipeline. The type is determined
            by the generic parameter ``T`` of the concrete subclass.
        """
