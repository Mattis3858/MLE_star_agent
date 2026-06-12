"""LLM clients, retries, and C5 token accounting.

All calls go through call_reasoner/call_coder so retries and budget
enforcement apply uniformly. The ledger raises BudgetExceeded when the
cumulative token spend crosses the configured ceiling -> the search loop
terminates gracefully instead of bleeding tokens.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, wait_fixed

from . import config

load_dotenv()


class BudgetExceeded(RuntimeError):
    pass


class TokenLedger:
    def __init__(self, budget: int = config.TOKEN_BUDGET):
        self.budget = budget
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, response, prompt: str) -> None:
        self.calls += 1
        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))
        else:  # crude fallback: ~4 chars/token
            self.input_tokens += len(prompt) // 4
            self.output_tokens += len(getattr(response, "content", "")) // 4

    def check(self) -> None:
        if self.total > self.budget:
            raise BudgetExceeded(
                f"token budget exhausted: {self.total:,} > {self.budget:,}"
            )

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total": self.total,
            "budget": self.budget,
        }


LEDGER = TokenLedger()

_clients: dict = {}


def _client(model: str) -> ChatOllama:
    if model not in _clients:
        headers = {"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY', '')}"}
        _clients[model] = ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            client_kwargs={"headers": headers},
            temperature=0.0,
            keep_alive="60m",
        )
    return _clients[model]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def _invoke(model: str, prompt: str) -> str:
    LEDGER.check()
    resp = _client(model).invoke([HumanMessage(content=prompt)])
    LEDGER.record(resp, prompt)
    return resp.content


def call_reasoner(prompt: str) -> str:
    return _invoke(config.REASONER_MODEL, prompt)


def call_coder(prompt: str) -> str:
    return _invoke(config.CODER_MODEL, prompt)


def strip_fences(text: str) -> str:
    """LLMs are told to emit raw Python; this is the safety net."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.index("\n") if "\n" in t else len(t)
        t = t[first_nl + 1:]
    if t.endswith("```"):
        t = t[: t.rfind("```")]
    return t.replace("```python", "").replace("```", "").strip()
