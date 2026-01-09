from __future__ import annotations
import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str
    source_type: str = "material"

def _clean(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_pages(
    pages: list[dict],
    source: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    source_type: str = "material",
) -> list[Chunk]:
    docs: list[Document] = []
    for page in pages:
        text = _clean(page.get("text", ""))
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": source, "page": page.get("page"), "source_type": source_type},
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_docs = splitter.split_documents(docs)

    chunks: list[Chunk] = []
    for idx, doc in enumerate(split_docs, start=1):
        meta = doc.metadata or {}
        page = meta.get("page", 0)
        cid = f"{source}:p{page}:c{idx}"
        chunks.append(
            Chunk(
                text=doc.page_content,
                source=meta.get("source", source),
                page=page,
                chunk_id=cid,
                source_type=meta.get("source_type", source_type),
            )
        )
    return chunks
