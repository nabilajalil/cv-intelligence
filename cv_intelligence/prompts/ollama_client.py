
"""Local Ollama client for generating LLM responses."""

import json
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b-instruct"


class OllamaClientError(Exception):
    """Raised when the local Ollama request fails."""


def generate_local_response(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """Return raw model text without trusting its claims."""

    if not messages:
        raise OllamaClientError("Messages cannot be empty")

    payload = {
        "model": model,
        "messages": [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
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

    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)

        raw_response = result["message"]["content"]

        if not isinstance(raw_response, str):
            raise OllamaClientError(
                "Ollama returned an invalid response"
            )

        return raw_response

    except (HTTPError, URLError, TimeoutError) as exc:
        raise OllamaClientError(
            "Cannot communicate with local Ollama"
        ) from exc

    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaClientError(
            "Ollama returned an unexpected response"
        ) from exc
