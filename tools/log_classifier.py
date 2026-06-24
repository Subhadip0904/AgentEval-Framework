import yaml
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_llm = _cfg["llm"]

client = OpenAI(
    api_key=_llm.get("api_key", "ollama"),
    base_url=_llm.get("base_url", "http://localhost:11434/v1"),
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
        {
            "role": "system",
            "content": (
                "You are an expert SSD/NAND firmware engineer. "
                "Classify the given log entry strictly using this format:\n"
                "SEVERITY: [INFO|WARNING|CRITICAL]\n"
                "REASON: [one sentence explaining why]\n"
                "ACTION: [what the engineer should do next]\n"
                "Do not add any other text."
            ),
        },
        {"role": "user", "content": "Log: NVMe queue depth at 80% capacity"},
        {
            "role": "assistant",
            "content": (
                "SEVERITY: WARNING\n"
                "REASON: High queue depth indicates heavy I/O load approaching saturation.\n"
                "ACTION: Monitor for further increase and consider throttling write operations."
            ),
        },
        {"role": "user", "content": "Log: SMART self-test completed, no errors"},
        {
            "role": "assistant",
            "content": (
                "SEVERITY: INFO\n"
                "REASON: Routine self-test passed with no issues detected.\n"
                "ACTION: No action required."
            ),
        },
        {"role": "user", "content": "Log: Controller temperature exceeded 85 C threshold"},
        {
            "role": "assistant",
            "content": (
                "SEVERITY: CRITICAL\n"
                "REASON: Thermal threshold exceeded; sustained overheating risks data loss and hardware damage.\n"
                "ACTION: Immediately reduce workload and verify cooling system."
            ),
        },
        {"role": "user", "content": f"Log: {log_entry}"},
    ]

    response = client.chat.completions.create(
        model=_llm["model"],
        messages=messages,
        temperature=_llm.get("temperature", 0.0),
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    tests = [
        "Uncorrectable ECC error on block 0x3F2A, wear count 98%",
        "Write buffer flushed successfully",
        "PCIe link training failed, retrying...",
    ]
    for test in tests:
        print(f"\nInput: {test}")
        print(log_classifier.invoke({"log_entry": test}))
