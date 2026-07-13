from __future__ import annotations

from coola.equality import objects_are_equal

from glyphik.pipelines import BasePipeline
from glyphik.pipelines.factory import BasePipelineFactory, PipelineFactory


class SumPipeline(BasePipeline[int]):
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def execute(self) -> int:
        return sum(self._values)


def _make_pipeline() -> SumPipeline:
    """Return an InMemoryPipeline instance for testing."""
    return SumPipeline([1, 2, 3])


#####################################
#     Tests for PipelineFactory     #
#####################################


# --- Inheritance ---


def test_pipeline_factory_is_base_pipeline_factory() -> None:
    assert isinstance(PipelineFactory(_make_pipeline()), BasePipelineFactory)


# --- make_pipeline ---


def test_pipeline_factory_make_pipeline_returns_base_pipeline() -> None:
    factory = PipelineFactory(_make_pipeline())
    assert isinstance(factory.make_pipeline(), BasePipeline)


def test_pipeline_factory_make_pipeline_returns_same_instance() -> None:
    pipeline = _make_pipeline()
    factory = PipelineFactory(pipeline)
    assert factory.make_pipeline() is pipeline


def test_pipeline_factory_make_pipeline_returns_same_instance_across_calls() -> None:
    pipeline = _make_pipeline()
    factory = PipelineFactory(pipeline)
    assert factory.make_pipeline() is factory.make_pipeline()


# --- _get_repr_kwargs ---


def test_pipeline_factory_get_repr_kwargs() -> None:
    pipeline = _make_pipeline()
    factory = PipelineFactory(pipeline)
    assert objects_are_equal(factory._get_repr_kwargs(), {"pipeline": pipeline})


# --- __repr__ and __str__ ---


def test_pipeline_factory_repr_starts_with_class_name() -> None:
    factory = PipelineFactory(_make_pipeline())
    assert repr(factory).startswith("PipelineFactory(")


def test_pipeline_factory_str_starts_with_class_name() -> None:
    factory = PipelineFactory(_make_pipeline())
    assert str(factory).startswith("PipelineFactory(")


def test_pipeline_factory_repr_contains_pipeline() -> None:
    factory = PipelineFactory(_make_pipeline())
    assert "pipeline" in repr(factory)


def test_pipeline_factory_str_contains_pipeline() -> None:
    factory = PipelineFactory(_make_pipeline())
    assert "pipeline" in str(factory)
