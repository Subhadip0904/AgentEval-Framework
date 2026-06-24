import re
import yaml
from pathlib import Path

from langchain.tools import tool
from pydantic import BaseModel

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_root = Path(__file__).resolve().parent.parent
_ret_cfg = _cfg["retrieval"]

_retriever = None
try:
    from tools.pdf_loader import PDFLoader
    from tools.retriever import FAISSRetriever

    _loader = PDFLoader(
        pdf_dir=str(_root / "data" / "pdfs"),
        chunk_size=512,
        chunk_overlap=50,
    )
    _docs = _loader.load_pdfs()

    _retriever = FAISSRetriever(
        embedding_model=_ret_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        index_path=str(_root / _ret_cfg.get("index_path", "data/faiss_index")),
    )
    _retriever.build_index(_docs)
    print("[spec_search] FAISS index built successfully.")
except Exception as exc:
    print(f"[spec_search] FAISS unavailable ({exc}), using keyword fallback.")

_FALLBACK_DOCS = [
    {
        "text": "NVMe supports up to 65535 I/O queues with 65535 commands per queue, enabling massive flash parallelism.",
        "source": "NVMe_2.0_spec",
        "section": "3.1 Queue Model",
    },
    {
        "text": "The FTL maps logical block addresses to physical NAND, handles wear leveling, garbage collection, and bad block management.",
        "source": "SSD_Architecture_Guide",
        "section": "FTL Overview",
    },
    {
        "text": "PCIe 4.0 gives approximately 2 GB/s per lane. NVMe SSDs use x4 for 8 GB/s total bandwidth.",
        "source": "PCIe_4.0_spec",
        "section": "Bandwidth",
    },
    {
        "text": "UFS 3.1 supports full-duplex at 23.2 Gbps per lane over M-PHY physical layer.",
        "source": "JEDEC_UFS_3.1",
        "section": "Physical Layer",
    },
    {
        "text": "ECC detects and corrects bit errors in NAND flash. Uncorrectable ECC errors indicate block failure.",
        "source": "NAND_Flash_Guide",
        "section": "ECC",
    },
    {
        "text": "NAND flash wear leveling distributes write operations evenly across blocks to extend device lifespan.",
        "source": "SSD_Architecture_Guide",
        "section": "Wear Leveling",
    },
    {
        "text": "Garbage collection in SSD reclaims invalid pages by copying valid data to new blocks before erasing old ones.",
        "source": "SSD_Architecture_Guide",
        "section": "Garbage Collection",
    },
    {
        "text": "NVMe over Fabrics extends the NVMe protocol over network fabrics like RDMA and Fibre Channel.",
        "source": "NVMe_2.0_spec",
        "section": "NVMe-oF",
    },
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_retrieve(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Simple token-overlap fallback retrieval."""
    query_words = _tokens(query)
    scored = []
    for doc in _FALLBACK_DOCS:
        haystack = f"{doc['source']} {doc.get('section', '')} {doc['text']}"
        score = len(query_words & _tokens(haystack))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def _retrieve(query: str) -> list[dict[str, str]]:
    top_k = _ret_cfg.get("top_k", 3)
    if _retriever is not None:
        try:
            return _retriever.retrieve(query, top_k=top_k)
        except Exception as exc:
            print(f"[spec_search] FAISS retrieve failed ({exc}), using keyword fallback.")
    return _keyword_retrieve(query, top_k=top_k)


class SpecSearchInput(BaseModel):
    query: str


@tool("spec_search", args_schema=SpecSearchInput)
def spec_search(query: str) -> str:
    """
    Search SSD architecture specification documents including NVMe, PCIe,
    UFS, eMMC, NAND flash guides, and internal Micron architecture references.
    Use this when the engineer asks about specifications, protocols, standards,
    or architecture concepts. Returns relevant spec sections with source citations.
    Input: a natural language question about SSD/storage specifications.
    """
    results = _retrieve(query)
    if not results:
        return "No relevant sections found in indexed specifications."

    output = []
    for i, doc in enumerate(results, 1):
        source = doc.get("source", "unknown")
        section = doc.get("section", doc.get("page", ""))
        text = doc.get("text", "")
        output.append(f"[{i}] {source} - {section}\n{text}")
    return "\n\n".join(output)


if __name__ == "__main__":
    queries = [
        "How does NVMe queue depth work?",
        "What is wear leveling?",
        "ECC error correction in NAND",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        print(spec_search.invoke({"query": query}))
        print("-" * 60)
