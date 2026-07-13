from __future__ import annotations

import pytest

from glyphik.pipelines.base import BasePipeline
from glyphik.pipelines.resolve import resolve_pipeline

MINIMAL_PIPELINE_TARGET = "tests.unit.pipelines.test_resolve.MinimalPipeline"


class MinimalPipeline(BasePipeline):
    """Minimal concrete BasePipeline for testing."""

    def run(self) -> None:
        pass


######################################
#     Tests for resolve_pipeline     #
######################################


# --- Pass-through ---


def test_resolve_pipeline_returns_base_pipeline_instance() -> None:
    assert isinstance(resolve_pipeline(MinimalPipeline()), BasePipeline)


def test_resolve_pipeline_passthrough_returns_same_instance() -> None:
    pipeline = MinimalPipeline()
    assert resolve_pipeline(pipeline) is pipeline


# --- From dict ---


def test_resolve_pipeline_from_dict_returns_base_pipeline() -> None:
    result = resolve_pipeline({"_target_": MINIMAL_PIPELINE_TARGET})
    assert isinstance(result, BasePipeline)


def test_resolve_pipeline_from_dict_returns_correct_type() -> None:
    result = resolve_pipeline({"_target_": MINIMAL_PIPELINE_TARGET})
    assert isinstance(result, MinimalPipeline)


# --- Invalid input ---


def test_resolve_pipeline_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BasePipeline instance"):
        resolve_pipeline("not-a-pipeline")
