r"""Provide a concrete default factory for glyphik BasePipeline
models."""

from __future__ import annotations

__all__ = ["PipelineFactory"]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import MultilineDisplayMixin

from glyphik.pipelines.factory.base import BasePipelineFactory

if TYPE_CHECKING:
    from glyphik.pipelines.base import BasePipeline

T = TypeVar("T")


class PipelineFactory(BasePipelineFactory[T], MultilineDisplayMixin):
    """A concrete BasePipeline factory that wraps a pre-built
    :class:`~glyphik.pipelines.BasePipeline` instance.

    Use this when the pipeline is already instantiated and you
    simply want to wrap it in the :class:`~BasePipelineFactory`
    interface — for example, when injecting a fixed pipeline into a
    component that expects a factory.

    Args:
        pipeline: A fully configured
            :class:`~glyphik.pipelines.BasePipeline`
            instance to return from :meth:`make_pipeline`.

    Example:
        ```pycon
        >>> from glyphik.pipelines import BasePipeline
        >>> from glyphik.pipelines.factory import PipelineFactory
        >>> class SumPipeline(BasePipeline[int]):
        ...     def __init__(self, values: list[int]) -> None:
        ...         self._values = values
        ...     def run(self) -> int:
        ...         return sum(self._values)
        ...
        >>> factory = PipelineFactory(SumPipeline([1, 2, 3]))
        >>> pipeline = factory.make_pipeline()

        ```
    """

    def __init__(self, pipeline: BasePipeline[T]) -> None:
        self._pipeline = pipeline

    def make_pipeline(self) -> BasePipeline[T]:
        return self._pipeline

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"pipeline": self._pipeline}
