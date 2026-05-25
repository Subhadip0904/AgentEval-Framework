import requests

while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": "llama3.2:latest",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    answer = response.json()["choices"][0]["message"]["content"]

    print("\nAI:", answer)