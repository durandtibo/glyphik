from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from zenpyre.document_stores import InMemoryDocumentStore

from glyphik.data.sec import CompanyIdentifier
from glyphik.pipelines import CompanyDocumentAgentPipeline

AAPL = CompanyIdentifier(cik=320193, ticker="AAPL")
MSFT = CompanyIdentifier(cik=789019, ticker="MSFT")
GOOG = CompanyIdentifier(cik=1652044, ticker="GOOG")


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            id="1",
            page_content="aapl doc 1",
            metadata={"cik": AAPL.cik, "filing_date": "2023-01-01"},
        ),
        Document(
            id="2",
            page_content="aapl doc 2",
            metadata={"cik": AAPL.cik, "filing_date": "2022-01-01"},
        ),
        Document(
            id="3",
            page_content="msft doc 1",
            metadata={"cik": MSFT.cik, "filing_date": "2023-06-01"},
        ),
    ]


@pytest.fixture
def document_store(documents: list[Document]) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    store.add_documents(documents)
    return store


def _fake_agent() -> RunnableLambda:
    return RunnableLambda(
        lambda inp: {"company": inp["company"], "n_documents": len(inp["documents"])}
    )


def _fake_failing_agent(fail_on: CompanyIdentifier) -> RunnableLambda:
    """An agent that raises a ``ValueError`` for ``fail_on`` and returns
    a normal response for every other company."""

    def fn(inp: dict) -> dict:
        if inp["company"] == fail_on:
            msg = f"agent failed for {inp['company']}"
            raise ValueError(msg)
        return {"company": inp["company"], "n_documents": len(inp["documents"])}

    return RunnableLambda(fn)


######################################################
#     Tests for CompanyDocumentAgentPipeline         #
######################################################


# --- Constructor ---


def test_company_document_agent_pipeline_stores_companies(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._companies == [AAPL, MSFT]


def test_company_document_agent_pipeline_stores_document_store(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._document_store is document_store


def test_company_document_agent_pipeline_stores_agent(
    document_store: InMemoryDocumentStore,
) -> None:
    agent = _fake_agent()
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=agent
    )
    assert pipeline._agent is agent


def test_company_document_agent_pipeline_default_batch_size(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._batch_size == 0


def test_company_document_agent_pipeline_stores_batch_size(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent(), batch_size=4
    )
    assert pipeline._batch_size == 4


def test_company_document_agent_pipeline_negative_batch_size_raises(
    document_store: InMemoryDocumentStore,
) -> None:
    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        CompanyDocumentAgentPipeline(
            companies=[AAPL], document_store=document_store, agent=_fake_agent(), batch_size=-1
        )


def test_company_document_agent_pipeline_default_continue_on_error(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._continue_on_error is False


def test_company_document_agent_pipeline_stores_continue_on_error(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL],
        document_store=document_store,
        agent=_fake_agent(),
        continue_on_error=True,
    )
    assert pipeline._continue_on_error is True


def test_company_document_agent_pipeline_default_log_documents_metadata(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._log_documents_metadata is False


def test_company_document_agent_pipeline_stores_log_documents_metadata(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL],
        document_store=document_store,
        agent=_fake_agent(),
        log_documents_metadata=True,
    )
    assert pipeline._log_documents_metadata is True


def test_company_document_agent_pipeline_repr_contains_class_name(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    assert "CompanyDocumentAgentPipeline" in repr(pipeline)


def test_company_document_agent_pipeline_str_contains_class_name(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    assert "CompanyDocumentAgentPipeline" in str(pipeline)


# --- execute (sequential) ---


def test_company_document_agent_pipeline_execute_sequential_returns_list(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    result = pipeline.execute()
    assert isinstance(result, list)


def test_company_document_agent_pipeline_execute_sequential_returns_outputs(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    result = pipeline.execute()
    assert result == [
        {"company": AAPL, "n_documents": 2},
        {"company": MSFT, "n_documents": 1},
    ]


def test_company_document_agent_pipeline_execute_sequential_empty_companies(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    assert pipeline.execute() == []


# --- execute (batch) ---


def test_company_document_agent_pipeline_execute_batch_returns_list(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT], document_store=document_store, agent=_fake_agent(), batch_size=2
    )
    result = pipeline.execute()
    assert isinstance(result, list)


def test_company_document_agent_pipeline_execute_batch_returns_outputs(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT], document_store=document_store, agent=_fake_agent(), batch_size=2
    )
    result = pipeline.execute()
    assert result == [
        {"company": AAPL, "n_documents": 2},
        {"company": MSFT, "n_documents": 1},
    ]


def test_company_document_agent_pipeline_execute_batch_empty_companies(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[], document_store=document_store, agent=_fake_agent(), batch_size=2
    )
    assert pipeline.execute() == []


def test_company_document_agent_pipeline_execute_sequential_and_batch_agree(
    document_store: InMemoryDocumentStore,
) -> None:
    sequential = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    batched = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT], document_store=document_store, agent=_fake_agent(), batch_size=1
    )
    assert sequential.execute() == batched.execute()


# --- execute (continue_on_error, sequential) ---


def test_company_document_agent_pipeline_execute_sequential_raises_by_default(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT],
        document_store=document_store,
        agent=_fake_failing_agent(fail_on=AAPL),
        batch_size=0,
    )
    with pytest.raises(ValueError, match=r"agent failed for"):
        pipeline.execute()


def test_company_document_agent_pipeline_execute_sequential_continue_on_error_skips_failure(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT],
        document_store=document_store,
        agent=_fake_failing_agent(fail_on=AAPL),
        batch_size=0,
        continue_on_error=True,
    )
    result = pipeline.execute()
    assert result == [{"company": MSFT, "n_documents": 1}]


def test_company_document_agent_pipeline_execute_sequential_continue_on_error_last_fails(
    document_store: InMemoryDocumentStore,
) -> None:
    # A failure at the end of the sequence should not drop earlier,
    # already-succeeded outputs.
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT],
        document_store=document_store,
        agent=_fake_failing_agent(fail_on=MSFT),
        batch_size=0,
        continue_on_error=True,
    )
    result = pipeline.execute()
    assert result == [{"company": AAPL, "n_documents": 2}]


def test_company_document_agent_pipeline_execute_sequential_continue_on_error_no_failures(
    document_store: InMemoryDocumentStore,
) -> None:
    # When continue_on_error=True but nothing actually fails, behavior
    # should be identical to continue_on_error=False.
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT],
        document_store=document_store,
        agent=_fake_agent(),
        batch_size=0,
        continue_on_error=True,
    )
    result = pipeline.execute()
    assert result == [
        {"company": AAPL, "n_documents": 2},
        {"company": MSFT, "n_documents": 1},
    ]


# --- execute (continue_on_error, batch) ---


def test_company_document_agent_pipeline_execute_batch_raises_by_default(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT],
        document_store=document_store,
        agent=_fake_failing_agent(fail_on=AAPL),
        batch_size=2,
    )
    with pytest.raises(ValueError, match=r"agent failed for"):
        pipeline.execute()


def test_company_document_agent_pipeline_execute_batch_continue_on_error_skips_failure(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT],
        document_store=document_store,
        agent=_fake_failing_agent(fail_on=AAPL),
        batch_size=2,
        continue_on_error=True,
    )
    result = pipeline.execute()
    assert result == [{"company": MSFT, "n_documents": 1}]


def test_company_document_agent_pipeline_execute_batch_continue_on_error_no_failures(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL, MSFT],
        document_store=document_store,
        agent=_fake_agent(),
        batch_size=2,
        continue_on_error=True,
    )
    result = pipeline.execute()
    assert result == [
        {"company": AAPL, "n_documents": 2},
        {"company": MSFT, "n_documents": 1},
    ]


# --- _build_agent_input (log_documents_metadata) ---


def test_company_document_agent_pipeline_build_agent_input_log_documents_metadata_does_not_raise(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL],
        document_store=document_store,
        agent=_fake_agent(),
        log_documents_metadata=True,
    )
    # Just verify enabling the option does not raise, and that the
    # documents are still returned correctly (sorted by filing date).
    result = pipeline._build_agent_input(AAPL)
    assert result == {
        "company": AAPL,
        "documents": [
            Document(
                id="2",
                page_content="aapl doc 2",
                metadata={"cik": AAPL.cik, "filing_date": "2022-01-01"},
            ),
            Document(
                id="1",
                page_content="aapl doc 1",
                metadata={"cik": AAPL.cik, "filing_date": "2023-01-01"},
            ),
        ],
    }


def test_company_document_agent_pipeline_build_agent_input_log_documents_metadata_no_matching_documents(
    document_store: InMemoryDocumentStore,
) -> None:
    # Even with no matching documents, enabling the option should not
    # raise -- print_documents_metadata is called with an empty list.
    pipeline = CompanyDocumentAgentPipeline(
        companies=[GOOG],
        document_store=document_store,
        agent=_fake_agent(),
        log_documents_metadata=True,
    )
    result = pipeline._build_agent_input(GOOG)
    assert result == {"company": GOOG, "documents": []}


# --- _build_agent_input ---


def test_company_document_agent_pipeline_build_agent_input(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    result = pipeline._build_agent_input(AAPL)
    assert result == {
        "company": AAPL,
        "documents": [
            Document(
                id="2",
                page_content="aapl doc 2",
                metadata={"cik": AAPL.cik, "filing_date": "2022-01-01"},
            ),
            Document(
                id="1",
                page_content="aapl doc 1",
                metadata={"cik": AAPL.cik, "filing_date": "2023-01-01"},
            ),
        ],
    }


def test_company_document_agent_pipeline_build_agent_input_unknown_company(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = CompanyDocumentAgentPipeline(
        companies=[AAPL], document_store=document_store, agent=_fake_agent()
    )
    result = pipeline._build_agent_input(GOOG)
    assert result == {"company": GOOG, "documents": []}
