
"""Test the local Qwen model through Ollama's HTTP API."""

import json
from urllib.request import Request, urlopen

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:3b-instruct"


def main():
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a CV analysis assistant. "
                    "Return only valid JSON. "
                    "Do not invent candidate skills."
                ),
            },
            {
                "role": "user",
                "content": (
                    "A candidate's verified CV skills are "
                    "Python and Docker. "
                    "Return a JSON object with one key, "
                    "'skills', containing those two skills."
                ),
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        result = json.load(response)

    raw_answer = result["message"]["content"]
    parsed_answer = json.loads(raw_answer)

    print("Model:", result["model"])
    print("Qwen response:")
    print(json.dumps(parsed_answer, indent=2))


if __name__ == "__main__":
    main()
