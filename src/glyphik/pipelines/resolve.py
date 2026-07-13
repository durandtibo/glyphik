r"""Provide a resolution utility for creating glyphik BasePipeline
models."""

from __future__ import annotations

__all__ = ["resolve_pipeline"]

import logging
from typing import Any, TypeVar

from zenpyre.utils.resolve import resolve_object

from glyphik.pipelines.base import BasePipeline

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def resolve_pipeline[T](
    pipeline: BasePipeline[T] | dict[str, Any],
) -> BasePipeline[T]:
    """Resolve a :class:`~glyphik.pipelines.base.BasePipeline` instance
    from an existing object or a configuration dictionary.

    If ``pipeline`` is already a
    :class:`~glyphik.pipelines.base.BasePipeline` instance it is
    returned as-is.  If it is a :class:`dict`, it is treated as an
    ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        pipeline: Either a fully configured
            :class:`~glyphik.pipelines.base.BasePipeline`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured :class:`~glyphik.pipelines.base.BasePipeline`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~glyphik.pipelines.base.BasePipeline`
            instance.

    Example:
        ```pycon
        >>> from glyphik.pipelines.base import BasePipeline
        >>> from glyphik.pipelines.resolve import resolve_pipeline
        >>> class MyPipeline(BasePipeline):
        ...     def run(self) -> None:
        ...         pass
        ...
        >>> # From an existing instance:
        >>> pipeline = resolve_pipeline(MyPipeline())
        >>> # From a configuration dictionary:
        >>> pipeline = resolve_pipeline(  # doctest: +SKIP
        ...     {"_target_": "my_package.pipelines.MyPipeline"}
        ... )

        ```
    """
    return resolve_object(pipeline, cls=BasePipeline)
