from __future__ import annotations

from coola.equality import objects_are_equal

from glyphik.pipelines.base import BasePipeline
from glyphik.pipelines.factory import (
    BasePipelineFactory,
    ConfigurablePipelineFactory,
)
from tests.unit.pipelines.factory.test_vanilla import SumPipeline

PIPELINE_TARGET = "tests.unit.pipelines.factory.test_vanilla.SumPipeline"


def _make_pipeline() -> SumPipeline:
    """Return an SumPipeline instance for testing."""
    return SumPipeline([1, 2, 3])


##################################################
#     Tests for ConfigurablePipelineFactory      #
##################################################


# --- Inheritance ---


def test_configurable_pipeline_factory_is_base_pipeline_factory() -> None:
    assert isinstance(ConfigurablePipelineFactory(_make_pipeline()), BasePipelineFactory)


# --- make_pipeline from instance ---


def test_configurable_pipeline_factory_make_pipeline_returns_base_pipeline() -> None:
    factory = ConfigurablePipelineFactory(_make_pipeline())
    assert isinstance(factory.make_pipeline(), BasePipeline)


def test_configurable_pipeline_factory_make_pipeline_returns_same_instance() -> None:
    pipeline = _make_pipeline()
    factory = ConfigurablePipelineFactory(pipeline)
    assert factory.make_pipeline() is pipeline


# --- make_pipeline from dict ---


def test_configurable_pipeline_factory_make_pipeline_from_dict_returns_base_pipeline() -> None:
    factory = ConfigurablePipelineFactory({"_target_": PIPELINE_TARGET, "values": [1, 2, 3]})
    assert isinstance(factory.make_pipeline(), BasePipeline)


def test_configurable_pipeline_factory_make_pipeline_from_dict_returns_correct_type() -> None:
    factory = ConfigurablePipelineFactory({"_target_": PIPELINE_TARGET, "values": [1, 2, 3]})
    assert isinstance(factory.make_pipeline(), SumPipeline)


# --- _get_repr_kwargs ---


def test_configurable_pipeline_factory_get_repr_kwargs_instance() -> None:
    pipeline = _make_pipeline()
    factory = ConfigurablePipelineFactory(pipeline)
    assert objects_are_equal(factory._get_repr_kwargs(), {"pipeline": pipeline})


def test_configurable_pipeline_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": PIPELINE_TARGET, "values": [1, 2, 3]}
    factory = ConfigurablePipelineFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"pipeline": config})


# --- __repr__ and __str__ ---


def test_configurable_pipeline_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurablePipelineFactory(_make_pipeline())
    assert repr(factory).startswith("ConfigurablePipelineFactory(")


def test_configurable_pipeline_factory_str_starts_with_class_name() -> None:
    factory = ConfigurablePipelineFactory(_make_pipeline())
    assert str(factory).startswith("ConfigurablePipelineFactory(")


def test_configurable_pipeline_factory_repr_contains_pipeline() -> None:
    factory = ConfigurablePipelineFactory(_make_pipeline())
    assert "pipeline" in repr(factory)


def test_configurable_pipeline_factory_str_contains_pipeline() -> None:
    factory = ConfigurablePipelineFactory(_make_pipeline())
    assert "pipeline" in str(factory)
