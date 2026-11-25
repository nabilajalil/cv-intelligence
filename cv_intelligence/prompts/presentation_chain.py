"""LangChain inference for constrained presentation selection."""

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

from .presentation_prompt import PRESENTATION_PROMPT
from .response_validation import PresentationChoice
from .token_budget import MAX_CONTEXT_TOKENS


MAX_PRESENTATION_OUTPUT_TOKENS = 64
INFERENCE_TIMEOUT_SECONDS = 10


def parse_presentation(message):
    """Validate model output and preserve inference metadata."""

    choice = PresentationChoice.model_validate_json(
        message.content
    )

    return {
        "choice": choice,
        "usage_metadata": message.usage_metadata or {},
    }


def build_presentation_chain():
    """Construct the constrained presentation chain."""

    llm = ChatOllama(
        model="qwen2.5:3b-instruct",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        client_kwargs={
            "timeout": INFERENCE_TIMEOUT_SECONDS,
        },
        format=PresentationChoice.model_json_schema(),
        num_ctx=MAX_CONTEXT_TOKENS,
        num_predict=MAX_PRESENTATION_OUTPUT_TOKENS,
    )

    return (
        PRESENTATION_PROMPT
        | llm
        | RunnableLambda(parse_presentation)
    )
