r"""Provide the base factory interface for creating glyphik BasePipeline
models."""

from __future__ import annotations

__all__ = ["BasePipelineFactory"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from glyphik.pipelines.base import BasePipeline

T = TypeVar("T")


class BasePipelineFactory[T](ABC):
    """Abstract base class for
    :class:`~glyphik.pipelines.BasePipeline` factories.

    Subclasses implement :meth:`make_pipeline` to instantiate and
    return a configured
    :class:`~glyphik.pipelines.BasePipeline` object.  This
    pattern decouples pipeline creation from the rest of the
    codebase, making it easy to swap pipelines (e.g. file, web,
    database) without changing call sites.

    Example:
        ```pycon
        >>> from glyphik.pipelines import BasePipeline
        >>> from glyphik.pipelines.factory import BasePipelineFactory
        >>> class SumPipeline(BasePipeline[int]):
        ...     def __init__(self, values: list[int]) -> None:
        ...         self._values = values
        ...     def run(self, config=None) -> int:
        ...         return sum(self._values)
        ...
        >>> class MyPipelineFactory(BasePipelineFactory[list[int]]):
        ...     def make_pipeline(self) -> SumPipeline:
        ...         return SumPipeline([1, 2, 3])
        ...
        >>> factory = MyPipelineFactory()
        >>> pipeline = factory.make_pipeline()

        ```
    """

    @abstractmethod
    def make_pipeline(self) -> BasePipeline[T]:
        """Create and return a configured BasePipeline instance.

        Returns:
            A :class:`~glyphik.pipelines.BasePipeline`
            instance ready for use.
        """
