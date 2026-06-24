from pathlib import Path

import requests
import yaml

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_llm = _cfg["llm"]

while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    response = requests.post(
        f"{_llm.get('base_url', 'http://localhost:11434/v1')}/chat/completions",
        json={
            "model": _llm["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        timeout=60,
    )

    answer = response.json()["choices"][0]["message"]["content"]

    print("\nAI:", answer)
