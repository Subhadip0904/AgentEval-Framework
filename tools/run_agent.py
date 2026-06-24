import time
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import compiled_graph

queries = [
    "Classify this log: Uncorrectable ECC error on block 0x3F2A, wear count 98%",
    "How fast is UFS 3.1 and what protocol does it use?",
]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print("=" * 60)
    result = compiled_graph.invoke({"messages": [HumanMessage(content=query)]})
    print(f"ANSWER: {result['messages'][-1].content}")
    time.sleep(5)
