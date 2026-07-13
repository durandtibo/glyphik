from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from coola.equality import objects_are_equal

from glyphik.pipelines.factory import (
    BasePipelineFactory,
    SecDocumentSummarizationPipelineFactory,
)

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.pipelines.factory.sec_document_summarization"


def _make_factory(tmp_path: Path, **overrides: object) -> SecDocumentSummarizationPipelineFactory:
    """Return a SecDocumentSummarizationPipelineFactory for testing."""
    kwargs = {
        "companies": ["AAPL", "MSFT"],
        "agent_factory": MagicMock(),
        "base_dir": tmp_path,
    }
    kwargs.update(overrides)
    return SecDocumentSummarizationPipelineFactory(**kwargs)


###############################################################
#   Tests for SecDocumentSummarizationPipelineFactory     #
###############################################################


# --- Inheritance ---


def test_sec_document_summarization_pipeline_factory_is_base_pipeline_factory(
    tmp_path: Path,
) -> None:
    assert isinstance(_make_factory(tmp_path), BasePipelineFactory)


# --- __init__ argument normalization ---


def test_sec_document_summarization_pipeline_factory_stores_companies_as_list(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path, companies=("AAPL", "MSFT"))
    assert factory._companies == ["AAPL", "MSFT"]


def test_sec_document_summarization_pipeline_factory_sanitizes_base_dir_from_str(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path, base_dir=str(tmp_path))
    assert factory._base_dir == tmp_path


def test_sec_document_summarization_pipeline_factory_stores_agent_factory(
    tmp_path: Path,
) -> None:
    agent_factory = MagicMock()
    factory = _make_factory(tmp_path, agent_factory=agent_factory)
    assert factory._agent_factory is agent_factory


# --- make_pipeline wiring ---


def test_sec_document_summarization_pipeline_factory_make_pipeline_builds_agent(
    tmp_path: Path,
) -> None:
    agent_factory = MagicMock()
    factory = _make_factory(tmp_path, agent_factory=agent_factory)
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory"),
        patch(f"{MODULE}.CompanyDocumentAgentPipeline"),
    ):
        factory.make_pipeline()
        agent_factory.make_agent.assert_called_once_with()


def test_sec_document_summarization_pipeline_factory_make_pipeline_opens_read_only_store(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory") as mock_store_factory_cls,
        patch(f"{MODULE}.CompanyDocumentAgentPipeline"),
    ):
        factory.make_pipeline()
        mock_store_factory_cls.assert_called_once_with(base_dir=tmp_path, read_only=True)
        mock_store_factory_cls.return_value.make_document_store.assert_called_once_with()


def test_sec_document_summarization_pipeline_factory_make_pipeline_wires_pipeline(
    tmp_path: Path,
) -> None:
    agent_factory = MagicMock()
    factory = _make_factory(tmp_path, companies=["AAPL", "MSFT"], agent_factory=agent_factory)
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory") as mock_store_factory_cls,
        patch(f"{MODULE}.CompanyDocumentAgentPipeline") as mock_pipeline_cls,
    ):
        factory.make_pipeline()
        mock_pipeline_cls.assert_called_once_with(
            companies=["AAPL", "MSFT"],
            document_store=mock_store_factory_cls.return_value.make_document_store.return_value,
            agent=agent_factory.make_agent.return_value,
        )


def test_sec_document_summarization_pipeline_factory_make_pipeline_returns_pipeline(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory"),
        patch(f"{MODULE}.CompanyDocumentAgentPipeline") as mock_pipeline_cls,
    ):
        result = factory.make_pipeline()
        assert result is mock_pipeline_cls.return_value


# --- _get_repr_kwargs ---


def test_sec_document_summarization_pipeline_factory_get_repr_kwargs(tmp_path: Path) -> None:
    agent_factory = MagicMock()
    factory = _make_factory(tmp_path, companies=["AAPL", "MSFT"], agent_factory=agent_factory)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {"companies": ["AAPL", "MSFT"], "agent_factory": agent_factory, "base_dir": tmp_path},
    )


# --- __repr__ and __str__ ---


def test_sec_document_summarization_pipeline_factory_repr_starts_with_class_name(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    assert repr(factory).startswith("SecDocumentSummarizationPipelineFactory(")


def test_sec_document_summarization_pipeline_factory_str_starts_with_class_name(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    assert str(factory).startswith("SecDocumentSummarizationPipelineFactory(")
