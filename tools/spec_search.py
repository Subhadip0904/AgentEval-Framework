import os
from langchain.tools import tool
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from tools.retriever import FAISSRetriever
from tools.pdf_loader import PDFLoader

load_dotenv(r"C:\Users\User\Desktop\micronprep\.env")

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Initialize retriever
pdf_loader = PDFLoader()
retriever = FAISSRetriever(index_path="data/faiss_index")

# Load or build index
if not retriever.load_index():
    documents = pdf_loader.load_pdfs()
    retriever.build_index(documents)


def _retrieve(query: str, top_k: int = 3) -> list:
    """Retrieve documents using FAISS vector search"""
    try:
        return retriever.retrieve(query, top_k=top_k)
    except Exception as e:
        print(f"Error in retrieval: {e}")
        # Fallback to simple keyword matching
        docs = pdf_loader.load_pdfs()
        query_words = set(query.lower().split())
        scored = []
        for doc in docs:
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
    UFS, eMMC, and internal Micron architecture guides using semantic search.
    Use this when the engineer asks about specifications, protocols, or standards.
    Input: a natural language question. Output: relevant spec sections with citations.
    """
    results = _retrieve(query)
    if not results:
        return "No relevant sections found in indexed specifications."
    output = []
    for i, doc in enumerate(results):
        source = doc.get("source", "Unknown")
        page = doc.get("page", "?")
        text = doc.get("text", "")
        output.append(f"[{i+1}] {source} (page {page})\n{text[:300]}...")
    return "\n\n".join(output)

if __name__ == "__main__":
    result = spec_search.invoke({"query": "How does NVMe queue depth work?"})
    print(result)