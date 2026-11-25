"""LangChain generation chain for candidate response polishing."""

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ConfigDict, Field

from .response_polishing import RESPONSE_POLISHING_PROMPT
from .token_budget import MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS


class PolishedResponse(BaseModel):
    """Expected structured response from the local LLM."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=2000)


def build_response_chain():
    """Construct the prompt, model and output parsing chain."""

    llm = ChatOllama(
        model="qwen2.5:3b-instruct",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        format="json",
        num_ctx=MAX_CONTEXT_TOKENS,
        num_predict=MAX_OUTPUT_TOKENS,
    )

    parser = PydanticOutputParser(
        pydantic_object=PolishedResponse
    )

    return RESPONSE_POLISHING_PROMPT | llm | parser
