import os
from langchain.tools import tool
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(r"C:\Users\User\Desktop\micronprep\.env")

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


class CodeExplainerInput(BaseModel):
    code: str
    language: str = "c"  # c, verilog, python, etc.
    context: str = ""


@tool("code_explainer", args_schema=CodeExplainerInput)
def code_explainer(code: str, language: str = "c", context: str = "") -> str:
    """
    Explain firmware code snippets (C, Verilog, etc.) in the context of SSD architecture.
    Use this when an engineer provides code and wants to understand its purpose or behavior.
    Input: code snippet, programming language, and optional context.
    Output: Explanation including purpose, key operations, and hardware implications.
    """
    system_prompt = f"""You are an expert SSD firmware engineer. Explain the following {language} code snippet.
Focus on:
1. Purpose of the code
2. Key operations and their hardware implications
3. Performance or reliability considerations
4. Any potential issues or optimizations

Be concise but technical."""

    if context:
        system_prompt += f"\n\nContext: {context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Explain this code:\n\n```{language}\n{code}\n```"}
    ]

    response = client.chat.completions.create(
        model="llama3.2",
        messages=messages,
        temperature=0.0,
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Example C code for ECC syndrome calculation
    example_code = """
    uint8_t calculate_ecc_syndrome(uint8_t *data, int len) {
        uint8_t syndrome = 0;
        for (int i = 0; i < len; i++) {
            syndrome ^= data[i];
        }
        return syndrome;
    }
    """
    result = code_explainer.invoke({
        "code": example_code,
        "language": "c",
        "context": "ECC error detection in NAND controller"
    })
    print(result)
