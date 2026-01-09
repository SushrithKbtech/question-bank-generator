from __future__ import annotations
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

def get_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=model_name)

def get_client(persist_dir: str = "data/vector_db") -> str:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return persist_dir

def get_collection(persist_dir: str, name: str, embedder: SentenceTransformerEmbeddings):
    return Chroma(
        collection_name=name,
        embedding_function=embedder,
        persist_directory=persist_dir,
    )

def upsert_chunks(collection, chunks, batch_size: int = 64):
    texts = [c.text for c in chunks]
    metas = [{"source": c.source, "page": c.page, "source_type": c.source_type} for c in chunks]
    ids = [c.chunk_id for c in chunks]

    for i in range(0, len(chunks), batch_size):
        b_texts = texts[i:i+batch_size]
        b_metas = metas[i:i+batch_size]
        b_ids = ids[i:i+batch_size]
        collection.add_texts(texts=b_texts, metadatas=b_metas, ids=b_ids)
    collection.persist()
