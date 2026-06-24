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


class CodeInput(BaseModel):
    code_snippet: str
    language: str = "C"


@tool("code_explainer", args_schema=CodeInput)
def code_explainer(code_snippet: str, language: str = "C") -> str:
    """
    Explain firmware or hardware-related code snippets in plain English.
    Use this when an engineer pastes a code fragment and asks what it does,
    or wants to understand register manipulations, memory-mapped I/O,
    interrupt handlers, or driver logic.
    Input: code_snippet (the code to explain), language (e.g. C, Python, Verilog, SystemVerilog).
    Output: a plain-English explanation of what the code does.
    """
    response = client.chat.completions.create(
        model=_llm["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an expert in {language} firmware and hardware engineering, "
                    "specializing in SSD controllers, NAND flash, PCIe/NVMe subsystems, "
                    "and embedded systems. When given a code snippet, explain:\n"
                    "1. What the code does at a high level (1-2 sentences).\n"
                    "2. Any notable hardware interactions or register operations.\n"
                    "3. Potential edge cases or bugs, if any.\n"
                    "Be concise and technical."
                ),
            },
            {"role": "user", "content": f"Explain this {language} code:\n\n{code_snippet}"},
        ],
        temperature=_llm.get("temperature", 0.0),
        max_tokens=_llm.get("max_tokens", 500),
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    sample = """
    void nvme_submit_cmd(struct nvme_queue *nvmeq, struct nvme_command *cmd) {
        u16 tail = nvmeq->sq_tail;
        memcpy(&nvmeq->sq_cmds[tail], cmd, sizeof(*cmd));
        if (++tail == nvmeq->q_depth)
            tail = 0;
        writel(tail, nvmeq->q_db);
        nvmeq->sq_tail = tail;
    }
    """
    print(code_explainer.invoke({"code_snippet": sample, "language": "C"}))
