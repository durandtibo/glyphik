from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from coola.equality import objects_are_equal
from zenpyre.agents.factory import BaseAgentFactory

from glyphik.pipelines.factory import (
    BasePipelineFactory,
    SecDocumentSummarizationPipelineFactory,
)

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.pipelines.factory.sec_document_summarization"


def _make_agent_factory() -> MagicMock:
    """Return a MagicMock that passes ``isinstance(...,
    BaseAgentFactory)`` checks."""
    return MagicMock(spec=BaseAgentFactory)


def _make_factory(tmp_path: Path, **overrides: object) -> SecDocumentSummarizationPipelineFactory:
    """Return a SecDocumentSummarizationPipelineFactory for testing."""
    kwargs = {
        "companies": ["AAPL", "MSFT"],
        "agent_factory": _make_agent_factory(),
        "base_dir": tmp_path,
    }
    kwargs.update(overrides)
    return SecDocumentSummarizationPipelineFactory(**kwargs)


###########################################################
#   Tests for SecDocumentSummarizationPipelineFactory     #
###########################################################


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
    agent_factory = _make_agent_factory()
    factory = _make_factory(tmp_path, agent_factory=agent_factory)
    assert factory._agent_factory is agent_factory


def test_sec_document_summarization_pipeline_factory_resolves_agent_factory_from_dict(
    tmp_path: Path,
) -> None:
    inner_agent_factory = _make_agent_factory()
    with patch("zenpyre.utils.resolve.factory", return_value=inner_agent_factory) as mock_factory:
        factory = _make_factory(
            tmp_path,
            agent_factory={"_target_": "some.AgentFactory", "arg": 1},
        )
    mock_factory.assert_called_once_with(_target_="some.AgentFactory", arg=1)
    assert factory._agent_factory is inner_agent_factory


def test_sec_document_summarization_pipeline_factory_invalid_agent_factory_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="Received object is not a BaseAgentFactory instance"):
        _make_factory(tmp_path, agent_factory="not an agent factory")


def test_sec_document_summarization_pipeline_factory_default_batch_size(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path)
    assert factory._batch_size == 0


def test_sec_document_summarization_pipeline_factory_stores_batch_size(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, batch_size=8)
    assert factory._batch_size == 8


def test_sec_document_summarization_pipeline_factory_default_config(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path)
    assert factory._config is None


def test_sec_document_summarization_pipeline_factory_stores_config(tmp_path: Path) -> None:
    config = {"tags": ["summarization"]}
    factory = _make_factory(tmp_path, config=config)
    assert factory._config is config


def test_sec_document_summarization_pipeline_factory_default_continue_on_error(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    assert factory._continue_on_error is False


def test_sec_document_summarization_pipeline_factory_stores_continue_on_error(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path, continue_on_error=True)
    assert factory._continue_on_error is True


def test_sec_document_summarization_pipeline_factory_default_log_documents_metadata(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    assert factory._log_documents_metadata is False


def test_sec_document_summarization_pipeline_factory_stores_log_documents_metadata(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path, log_documents_metadata=True)
    assert factory._log_documents_metadata is True


# --- make_pipeline wiring ---


def test_sec_document_summarization_pipeline_factory_make_pipeline_builds_agent(
    tmp_path: Path,
) -> None:
    agent_factory = _make_agent_factory()
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
    agent_factory = _make_agent_factory()
    config = {"tags": ["summarization"]}
    factory = _make_factory(
        tmp_path,
        companies=["AAPL", "MSFT"],
        agent_factory=agent_factory,
        batch_size=8,
        config=config,
        continue_on_error=True,
        log_documents_metadata=True,
    )
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory") as mock_store_factory_cls,
        patch(f"{MODULE}.CompanyDocumentAgentPipeline") as mock_pipeline_cls,
    ):
        factory.make_pipeline()
        mock_pipeline_cls.assert_called_once_with(
            companies=["AAPL", "MSFT"],
            document_store=mock_store_factory_cls.return_value.make_document_store.return_value,
            agent=agent_factory.make_agent.return_value,
            batch_size=8,
            config=config,
            continue_on_error=True,
            log_documents_metadata=True,
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
    agent_factory = _make_agent_factory()
    config = {"tags": ["summarization"]}
    factory = _make_factory(
        tmp_path,
        companies=["AAPL", "MSFT"],
        agent_factory=agent_factory,
        batch_size=8,
        config=config,
        continue_on_error=True,
        log_documents_metadata=True,
    )
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "companies": ["AAPL", "MSFT"],
            "agent_factory": agent_factory,
            "base_dir": tmp_path,
            "batch_size": 8,
            "config": config,
            "continue_on_error": True,
            "log_documents_metadata": True,
        },
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


# --- from_sp1500 ---


def test_sec_document_summarization_pipeline_factory_from_sp1500_resolves_path(
    tmp_path: Path,
) -> None:
    agent_factory = _make_agent_factory()
    with patch(f"{MODULE}.get_company_identifiers", return_value=[]) as mock_get_identifiers:
        SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=agent_factory, base_dir=tmp_path
        )
        mock_get_identifiers.assert_called_once_with(
            tmp_path / "SP1500" / "company_identifier.json"
        )


def test_sec_document_summarization_pipeline_factory_from_sp1500_sanitizes_base_dir_from_str(
    tmp_path: Path,
) -> None:
    agent_factory = _make_agent_factory()
    with patch(f"{MODULE}.get_company_identifiers", return_value=[]):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=agent_factory, base_dir=str(tmp_path)
        )
    assert factory._base_dir == tmp_path


def test_sec_document_summarization_pipeline_factory_from_sp1500_stores_agent_factory(
    tmp_path: Path,
) -> None:
    agent_factory = _make_agent_factory()
    with patch(f"{MODULE}.get_company_identifiers", return_value=[]):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=agent_factory, base_dir=tmp_path
        )
    assert factory._agent_factory is agent_factory


def test_sec_document_summarization_pipeline_factory_from_sp1500_all_companies_by_default(
    tmp_path: Path,
) -> None:
    companies = ["AAPL", "MSFT", "NVDA"]
    with patch(f"{MODULE}.get_company_identifiers", return_value=companies):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=_make_agent_factory(), base_dir=tmp_path
        )
    assert factory._companies == companies


def test_sec_document_summarization_pipeline_factory_from_sp1500_truncates_max_companies(
    tmp_path: Path,
) -> None:
    companies = ["AAPL", "MSFT", "NVDA"]
    with patch(f"{MODULE}.get_company_identifiers", return_value=companies):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=_make_agent_factory(), base_dir=tmp_path, max_companies=2
        )
    assert factory._companies == ["AAPL", "MSFT"]


def test_sec_document_summarization_pipeline_factory_from_sp1500_max_companies_larger_than_available(
    tmp_path: Path,
) -> None:
    companies = ["AAPL", "MSFT"]
    with patch(f"{MODULE}.get_company_identifiers", return_value=companies):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=_make_agent_factory(), base_dir=tmp_path, max_companies=10
        )
    assert factory._companies == ["AAPL", "MSFT"]


def test_sec_document_summarization_pipeline_factory_from_sp1500_returns_instance_of_class(
    tmp_path: Path,
) -> None:
    with patch(f"{MODULE}.get_company_identifiers", return_value=[]):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=_make_agent_factory(), base_dir=tmp_path
        )
    assert isinstance(factory, SecDocumentSummarizationPipelineFactory)


def test_sec_document_summarization_pipeline_factory_from_sp1500_default_extra_params(
    tmp_path: Path,
) -> None:
    with patch(f"{MODULE}.get_company_identifiers", return_value=[]):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=_make_agent_factory(), base_dir=tmp_path
        )
    assert factory._batch_size == 0
    assert factory._config is None
    assert factory._continue_on_error is False
    assert factory._log_documents_metadata is False


def test_sec_document_summarization_pipeline_factory_from_sp1500_forwards_extra_params(
    tmp_path: Path,
) -> None:
    config = {"tags": ["summarization"]}
    with patch(f"{MODULE}.get_company_identifiers", return_value=[]):
        factory = SecDocumentSummarizationPipelineFactory.from_sp1500(
            agent_factory=_make_agent_factory(),
            base_dir=tmp_path,
            batch_size=8,
            config=config,
            continue_on_error=True,
            log_documents_metadata=True,
        )
    assert factory._batch_size == 8
    assert factory._config is config
    assert factory._continue_on_error is True
    assert factory._log_documents_metadata is True
