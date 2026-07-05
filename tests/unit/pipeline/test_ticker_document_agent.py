from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from zenpyre.document_stores import InMemoryDocumentStore

from glyphik.pipeline import TickerDocumentAgentPipeline


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            id="1",
            page_content="aapl doc 1",
            metadata={"ticker": "AAPL", "filing_date": "2023-01-01"},
        ),
        Document(
            id="2",
            page_content="aapl doc 2",
            metadata={"ticker": "AAPL", "filing_date": "2022-01-01"},
        ),
        Document(
            id="3",
            page_content="msft doc 1",
            metadata={"ticker": "MSFT", "filing_date": "2023-06-01"},
        ),
    ]


@pytest.fixture
def document_store(documents: list[Document]) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    store.add_documents(documents)
    return store


def _fake_agent() -> RunnableLambda:
    return RunnableLambda(
        lambda inp: {"ticker": inp["ticker"], "n_documents": len(inp["documents"])}
    )


######################################################
#     Tests for TickerDocumentAgentPipeline          #
######################################################


# --- Constructor ---


def test_ticker_document_agent_pipeline_stores_tickers(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL", "MSFT"], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._tickers == ["AAPL", "MSFT"]


def test_ticker_document_agent_pipeline_stores_document_store(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._document_store is document_store


def test_ticker_document_agent_pipeline_stores_agent(document_store: InMemoryDocumentStore) -> None:
    agent = _fake_agent()
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=agent
    )
    assert pipeline._agent is agent


def test_ticker_document_agent_pipeline_default_batch_size(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=_fake_agent()
    )
    assert pipeline._batch_size == 0


def test_ticker_document_agent_pipeline_stores_batch_size(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=_fake_agent(), batch_size=4
    )
    assert pipeline._batch_size == 4


def test_ticker_document_agent_pipeline_negative_batch_size_raises(
    document_store: InMemoryDocumentStore,
) -> None:
    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        TickerDocumentAgentPipeline(
            tickers=["AAPL"], document_store=document_store, agent=_fake_agent(), batch_size=-1
        )


def test_ticker_document_agent_pipeline_repr_contains_class_name(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=_fake_agent()
    )
    assert "TickerDocumentAgentPipeline" in repr(pipeline)


def test_ticker_document_agent_pipeline_str_contains_class_name(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=_fake_agent()
    )
    assert "TickerDocumentAgentPipeline" in str(pipeline)


# --- execute (sequential) ---


def test_ticker_document_agent_pipeline_execute_sequential_returns_list(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL", "MSFT"], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    result = pipeline.execute()
    assert isinstance(result, list)


def test_ticker_document_agent_pipeline_execute_sequential_returns_outputs(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL", "MSFT"], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    result = pipeline.execute()
    assert result == [
        {"ticker": "AAPL", "n_documents": 2},
        {"ticker": "MSFT", "n_documents": 1},
    ]


def test_ticker_document_agent_pipeline_execute_sequential_empty_tickers(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=[], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    assert pipeline.execute() == []


# --- execute (batch) ---


def test_ticker_document_agent_pipeline_execute_batch_returns_list(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL", "MSFT"], document_store=document_store, agent=_fake_agent(), batch_size=2
    )
    result = pipeline.execute()
    assert isinstance(result, list)


def test_ticker_document_agent_pipeline_execute_batch_returns_outputs(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL", "MSFT"], document_store=document_store, agent=_fake_agent(), batch_size=2
    )
    result = pipeline.execute()
    assert result == [
        {"ticker": "AAPL", "n_documents": 2},
        {"ticker": "MSFT", "n_documents": 1},
    ]


def test_ticker_document_agent_pipeline_execute_batch_empty_tickers(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=[], document_store=document_store, agent=_fake_agent(), batch_size=2
    )
    assert pipeline.execute() == []


def test_ticker_document_agent_pipeline_execute_sequential_and_batch_agree(
    document_store: InMemoryDocumentStore,
) -> None:
    sequential = TickerDocumentAgentPipeline(
        tickers=["AAPL", "MSFT"], document_store=document_store, agent=_fake_agent(), batch_size=0
    )
    batched = TickerDocumentAgentPipeline(
        tickers=["AAPL", "MSFT"], document_store=document_store, agent=_fake_agent(), batch_size=1
    )
    assert sequential.execute() == batched.execute()


# --- _build_agent_input ---


def test_ticker_document_agent_pipeline_build_agent_input(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=_fake_agent()
    )
    result = pipeline._build_agent_input("AAPL")
    assert result == {
        "ticker": "AAPL",
        "documents": [
            Document(
                id="2",
                page_content="aapl doc 2",
                metadata={"ticker": "AAPL", "filing_date": "2022-01-01"},
            ),
            Document(
                id="1",
                page_content="aapl doc 1",
                metadata={"ticker": "AAPL", "filing_date": "2023-01-01"},
            ),
        ],
    }


def test_ticker_document_agent_pipeline_build_agent_input_unknown_ticker(
    document_store: InMemoryDocumentStore,
) -> None:
    pipeline = TickerDocumentAgentPipeline(
        tickers=["AAPL"], document_store=document_store, agent=_fake_agent()
    )
    result = pipeline._build_agent_input("GOOG")
    assert result == {"ticker": "GOOG", "documents": []}
