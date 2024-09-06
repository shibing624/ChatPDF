import os
from time import time
from graphrag import GraphRAG
from graphrag import wrap_embedding_func_with_attrs
from graphrag import deepseek_chat_complete, gpt_4o_mini_complete, gpt_4o_complete, ollama_model_complete
from text2vec import SentenceModel

pwd_path = os.path.abspath(os.path.dirname(__file__))

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["OLLAMA_MODEL"] = "qwen2:72b"


def main():
    WORKING_DIR = "./graphrag_cache_ollama_sanguo_test"
    test_file = os.path.join(pwd_path, "data/三国演义.txt")
    with open(test_file, encoding="utf-8") as f:
        FAKE_TEXT = f.read()
    print("FAKE_TEXT length:", len(FAKE_TEXT), " top3:", FAKE_TEXT[:30])
    FAKE_TEXT = FAKE_TEXT[:4000]
    print("FAKE_TEXT length:", len(FAKE_TEXT), " top3:", FAKE_TEXT[:30])

    EMBED_MODEL = SentenceModel("shibing624/text2vec-base-multilingual")

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
        max_token_size=EMBED_MODEL.max_seq_length,
    )
    async def text2vec_embedding(texts: list[str]):
        return EMBED_MODEL.encode(texts, normalize_embeddings=True)

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        embedding_func=text2vec_embedding,
        best_model_func=ollama_model_complete,
        cheap_model_func=ollama_model_complete,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    qs = [
        "三国演义的中心思想是啥？",
        "黄巾军是怎么被打败的？",
        "三国演义是哪三国？",
        "刘备对曹操的感情变化？"
    ]

    for i in qs:
        print('\n\n', '-' * 42)
        print(i)
        print(rag.query(i))


if __name__ == "__main__":
    main()
