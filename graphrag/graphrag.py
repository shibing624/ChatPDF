import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, List, Union
from loguru import logger

from ._model import gpt_4o_complete, gpt_4o_mini_complete, openai_embedding
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    local_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"graphrag_cache_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    # graph mode
    enable_local: bool = True

    # text chunking
    chunk_token_size: int = 2000
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        # init dir
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # init storages
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )

        # init functions
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings: Union[str, List[str]]):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        response = await local_query(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.community_reports,
            self.text_chunks,
            param,
            asdict(self),
        )
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings: Union[str, List[str]]):
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = await self._prepare_new_docs(string_or_strings)
            if not new_docs:
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking
            inserting_chunks = await self._prepare_inserting_chunks(new_docs)
            if not inserting_chunks:
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            # ---------- extract/summary entity and upsert to graph
            await self._process_entities_and_clusters(inserting_chunks)

            # ---------- commit upsert and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _prepare_new_docs(self, string_or_strings):
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not new_docs:
            logger.warning(f"All docs are already in the storage")
        return new_docs

    async def _prepare_inserting_chunks(self, new_docs):
        inserting_chunks = {}
        for doc_key, doc in new_docs.items():
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_key,
                }
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }
            inserting_chunks.update(chunks)
        _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        if not inserting_chunks:
            logger.warning(f"All chunks are already in the storage")
        return inserting_chunks

    async def _process_entities_and_clusters(self, inserting_chunks):
        await self.community_reports.drop()
        logger.info("[Entity Extraction]...")
        maybe_new_kg = await extract_entities(
            inserting_chunks,
            knwoledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            global_config=asdict(self),
        )
        if maybe_new_kg is None:
            logger.warning("No new entities found")
            return
        self.chunk_entity_relation_graph = maybe_new_kg
        logger.info("[Community Report]...")
        await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)
        await generate_community_report(self.community_reports, self.chunk_entity_relation_graph, asdict(self))

    async def _insert_done(self):
        tasks = [cast(StorageNameSpace, storage_inst).index_done_callback() for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunk_entity_relation_graph,
        ] if storage_inst is not None]
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = [cast(StorageNameSpace, storage_inst).index_done_callback() for storage_inst in [
            self.llm_response_cache
        ] if storage_inst is not None]
        await asyncio.gather(*tasks)
