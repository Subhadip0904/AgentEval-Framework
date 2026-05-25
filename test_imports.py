import sys
print("Python:", sys.executable)

import openai
import langchain
import fastapi
import faiss

print("openai   :", openai.__version__)
print("langchain:", langchain.__version__)
print("fastapi  :", fastapi.__version__)
print("faiss    : OK")
print()
print("All good — ready for the interview!")