import os
from langchain.tools import tool
from langchain_core.documents import Document
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(r"C:\Users\User\Desktop\micronprep\.env")

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# In-memory doc store — no FAISS, no torch needed
DOCS = [
    {"text": "NVMe supports up to 65535 I/O queues with 65535 commands per queue, enabling massive flash parallelism.", "source": "NVMe_2.0_spec", "section": "3.1 Queue Model"},
    {"text": "The FTL maps logical block addresses to physical NAND, handles wear levelling, GC, and bad block management.", "source": "SSD_Architecture_Guide", "section": "FTL Overview"},
    {"text": "PCIe 4.0 gives ~2 GB/s per lane. NVMe SSDs use x4 for 8 GB/s total bandwidth.", "source": "PCIe_4.0_spec", "section": "Bandwidth"},
    {"text": "UFS 3.1 supports full-duplex at 23.2 Gbps per lane over M-PHY physical layer.", "source": "JEDEC_UFS_3.1", "section": "Physical Layer"},
    {"text": "ECC detects and corrects bit errors in NAND flash. Uncorrectable ECC errors indicate block failure.", "source": "NAND_Flash_Guide", "section": "ECC"},
]

def _retrieve(query: str, top_k: int = 3) -> list:
    query_words = set(query.lower().split())
    scored = []
    for doc in DOCS:
        score = len(query_words & set(doc["text"].lower().split()))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

class SpecSearchInput(BaseModel):
    query: str

@tool("spec_search", args_schema=SpecSearchInput)
def spec_search(query: str) -> str:
    """
    Search SSD architecture specification documents including NVMe, PCIe,
    UFS, eMMC, and internal Micron architecture guides.
    Use this when the engineer asks about specifications, protocols, or standards.
    Input: a natural language question. Output: relevant spec sections with citations.
    """
    results = _retrieve(query)
    if not results:
        return "No relevant sections found in indexed specifications."
    output = []
    for i, doc in enumerate(results):
        output.append(f"[{i+1}] {doc['source']} — {doc['section']}\n{doc['text']}")
    return "\n\n".join(output)

if __name__ == "__main__":
    result = spec_search.invoke({"query": "How does NVMe queue depth work?"})
    print(result)