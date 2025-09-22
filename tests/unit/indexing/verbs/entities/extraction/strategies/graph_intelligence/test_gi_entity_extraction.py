# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
import unittest

from graphrag.index.operations.extract_graph.graph_extractor import (
    DEFAULT_ENTITY_TYPES,
)
from graphrag.index.operations.extract_graph.graph_intelligence_strategy import (
    run_extract_graph,
)
from graphrag.index.operations.extract_graph.typing import (
    Document,
)
from graphrag.language_model.response.base import BaseModelOutput, BaseModelResponse
from tests.unit.indexing.verbs.helpers.mock_llm import create_mock_llm


BASIC_GRAPH_RESPONSE = """
("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
##
("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 is related to TEST_ENTITY_1)
##
("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related<|>1))
""".strip()


class RecordingChatModel:
    def __init__(self, responses: list[str]):
        self._responses = responses
        self._response_index = 0
        self.prompts: list[str] = []

    async def achat(self, prompt: str, history: list | None = None, **kwargs):
        self.prompts.append(prompt)
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
        else:
            response = ""

        new_history = list(history) if history else []
        new_history.append({"prompt": prompt, "response": response})

        return BaseModelResponse(
            output=BaseModelOutput(content=response),
            history=new_history,
        )


class TestRunChain(unittest.IsolatedAsyncioTestCase):
    async def test_run_extract_graph_single_document_correct_entities_returned(self):
        results = await run_extract_graph(
            docs=[Document("test_text", "1")],
            entity_types=["person"],
            args={
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            model=create_mock_llm(
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip()
                ],
                name="test_run_extract_graph_single_document_correct_entities_returned",
            ),
        )

        # self.assertItemsEqual isn't available yet, or I am just silly
        # so we sort the lists and compare them
        assert sorted(["TEST_ENTITY_1", "TEST_ENTITY_2", "TEST_ENTITY_3"]) == sorted([
            entity["title"] for entity in results.entities
        ])

    async def test_run_extract_graph_multiple_documents_correct_entities_returned(
        self,
    ):
        results = await run_extract_graph(
            docs=[Document("text_1", "1"), Document("text_2", "2")],
            entity_types=["person"],
            args={
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            model=create_mock_llm(
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    """.strip(),
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip(),
                ],
                name="test_run_extract_graph_multiple_documents_correct_entities_returned",
            ),
        )

        # self.assertItemsEqual isn't available yet, or I am just silly
        # so we sort the lists and compare them
        assert sorted(["TEST_ENTITY_1", "TEST_ENTITY_2", "TEST_ENTITY_3"]) == sorted([
            entity["title"] for entity in results.entities
        ])

    async def test_run_extract_graph_multiple_documents_correct_edges_returned(self):
        results = await run_extract_graph(
            docs=[Document("text_1", "1"), Document("text_2", "2")],
            entity_types=["person"],
            args={
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            model=create_mock_llm(
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    """.strip(),
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip(),
                ],
                name="test_run_extract_graph_multiple_documents_correct_edges_returned",
            ),
        )

        # self.assertItemsEqual isn't available yet, or I am just silly
        # so we sort the lists and compare them
        graph = results.graph
        assert graph is not None, "No graph returned!"

        # convert to strings for more visual comparison
        edges_str = sorted([f"{edge[0]} -> {edge[1]}" for edge in graph.edges])
        assert edges_str == sorted([
            "TEST_ENTITY_1 -> TEST_ENTITY_2",
            "TEST_ENTITY_1 -> TEST_ENTITY_3",
        ])

    async def test_run_extract_graph_multiple_documents_correct_entity_source_ids_mapped(
        self,
    ):
        results = await run_extract_graph(
            docs=[Document("text_1", "1"), Document("text_2", "2")],
            entity_types=["person"],
            args={
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            model=create_mock_llm(
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    """.strip(),
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip(),
                ],
                name="test_run_extract_graph_multiple_documents_correct_entity_source_ids_mapped",
            ),
        )

        graph = results.graph
        assert graph is not None, "No graph returned!"

        # TODO: The edges might come back in any order, but we're assuming they're coming
        # back in the order that we passed in the docs, that might not be true
        assert (
            graph.nodes["TEST_ENTITY_3"].get("source_id") == "2"
        )  # TEST_ENTITY_3 should be in just 2
        assert (
            graph.nodes["TEST_ENTITY_2"].get("source_id") == "1"
        )  # TEST_ENTITY_2 should be in just 1
        ids_str = graph.nodes["TEST_ENTITY_1"].get("source_id") or ""
        assert sorted(ids_str.split(",")) == sorted([
            "1",
            "2",
        ])  # TEST_ENTITY_1 should be 1 and 2

    async def test_run_extract_graph_multiple_documents_correct_edge_source_ids_mapped(
        self,
    ):
        results = await run_extract_graph(
            docs=[Document("text_1", "1"), Document("text_2", "2")],
            entity_types=["person"],
            args={
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            model=create_mock_llm(
                responses=[
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_2<|>COMPANY<|>TEST_ENTITY_2 owns TEST_ENTITY_1 and also shares an address with TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_2<|>TEST_ENTITY_1 and TEST_ENTITY_2 are related because TEST_ENTITY_1 is 100% owned by TEST_ENTITY_2 and the two companies also share the same address)<|>2)
                    ##
                    """.strip(),
                    """
                    ("entity"<|>TEST_ENTITY_1<|>COMPANY<|>TEST_ENTITY_1 is a test company)
                    ##
                    ("entity"<|>TEST_ENTITY_3<|>PERSON<|>TEST_ENTITY_3 is director of TEST_ENTITY_1)
                    ##
                    ("relationship"<|>TEST_ENTITY_1<|>TEST_ENTITY_3<|>TEST_ENTITY_1 and TEST_ENTITY_3 are related because TEST_ENTITY_3 is director of TEST_ENTITY_1<|>1))
                    """.strip(),
                ],
                name="test_run_extract_graph_multiple_documents_correct_edge_source_ids_mapped",
            ),
        )

        graph = results.graph
        assert graph is not None, "No graph returned!"
        edges = list(graph.edges(data=True))

        # should only have 2 edges
        assert len(edges) == 2

        # Sort by source_id for consistent ordering
        edge_source_ids = sorted([edge[2].get("source_id", "") for edge in edges])
        assert edge_source_ids[0].split(",") == ["1"]
        assert edge_source_ids[1].split(",") == ["2"]

    async def test_run_extract_graph_defaults_entity_types_when_missing(self):
        model = RecordingChatModel([BASIC_GRAPH_RESPONSE])

        results = await run_extract_graph(
            docs=[Document("text_1", "1")],
            entity_types=None,
            args={
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            model=model,
        )

        assert results.entities, "Expected entities to be returned"
        assert model.prompts, "Expected the model to receive at least one prompt"
        assert f"[{','.join(DEFAULT_ENTITY_TYPES)}]" in model.prompts[0]

    async def test_run_extract_graph_honors_custom_entity_types_override(self):
        custom_entity_types = ["alpha", "beta"]
        model = RecordingChatModel([BASIC_GRAPH_RESPONSE])

        await run_extract_graph(
            docs=[Document("text_1", "1")],
            entity_types=custom_entity_types,
            args={
                "max_gleanings": 0,
                "summarize_descriptions": False,
            },
            model=model,
        )

        assert model.prompts, "Expected the model to receive at least one prompt"
        assert f"[{','.join(custom_entity_types)}]" in model.prompts[0]
