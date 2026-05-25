import os
from langchain.tools import tool
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(r"C:\Users\User\Desktop\micronprep\.env")

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

class LogInput(BaseModel):
    log_entry: str

@tool("log_classifier", args_schema=LogInput)
def log_classifier(log_entry: str) -> str:
    """
    Classify a firmware or SSD controller log entry as INFO, WARNING, or CRITICAL.
    Use this when an engineer provides a log line and wants to know its severity.
    Input: a raw log entry string.
    Output: SEVERITY, REASON, and ACTION in structured format.
    """
    messages = [
        {"role": "system", "content": (
            "Classify SSD firmware log entries. "
            "Respond ONLY in this format:\n"
            "SEVERITY: [INFO|WARNING|CRITICAL]\n"
            "REASON: [one sentence]\n"
            "ACTION: [what engineer should do]"
        )},
        {"role": "user", "content": "Log: NVMe queue depth at 80% capacity"},
        {"role": "assistant", "content": "SEVERITY: WARNING\nREASON: High queue depth indicates heavy I/O load.\nACTION: Monitor for further increase."},
        {"role": "user", "content": "Log: SMART self-test completed, no errors"},
        {"role": "assistant", "content": "SEVERITY: INFO\nREASON: Routine test passed cleanly.\nACTION: No action required."},
        {"role": "user", "content": f"Log: {log_entry}"},
    ]
    response = client.chat.completions.create(
        model="llama3.2",
        messages=messages,
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(log_classifier.invoke({"log_entry": "Uncorrectable ECC error on block 0x3F2A, wear count 98%"}))
    print()
    print(log_classifier.invoke({"log_entry": "Write buffer flushed successfully"}))