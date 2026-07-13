r"""Provide a configurable factory for glyphik BasePipeline models."""

from __future__ import annotations

__all__ = ["ConfigurablePipelineFactory"]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import MultilineDisplayMixin

from glyphik.pipelines.factory.base import BasePipelineFactory
from glyphik.pipelines.resolve import resolve_pipeline

if TYPE_CHECKING:
    from glyphik.pipelines.base import BasePipeline

T = TypeVar("T")


class ConfigurablePipelineFactory(BasePipelineFactory[T], MultilineDisplayMixin):
    """A concrete BasePipeline factory that accepts either a pre-built
    :class:`~glyphik.pipelines.BasePipeline` instance or a
    configuration dictionary.

    When a dict is provided it is resolved at each :meth:`make_pipeline`
    call via :func:`~glyphik.pipelines.resolve.resolve_pipeline`, which
    uses ``objectory`` to instantiate the configured class.  When an
    instance is provided it is returned as-is.

    Args:
        pipeline: A fully configured
            :class:`~glyphik.pipelines.BasePipeline`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Example:
        ```pycon
        >>> from glyphik.pipelines import BasePipeline
        >>> from glyphik.pipelines.factory import ConfigurablePipelineFactory
        >>> class SumPipeline(BasePipeline[int]):
        ...     def __init__(self, values: list[int]) -> None:
        ...         self._values = values
        ...     def run(self) -> int:
        ...         return sum(self._values)
        ...
        >>> factory = ConfigurablePipelineFactory(SumPipeline([1, 2, 3]))
        >>> pipeline = factory.make_pipeline()

        ```
    """

    def __init__(self, pipeline: BasePipeline[T] | dict[str, Any]) -> None:
        self._pipeline = pipeline

    def make_pipeline(self) -> BasePipeline[T]:
        return resolve_pipeline(self._pipeline)

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"pipeline": self._pipeline}
