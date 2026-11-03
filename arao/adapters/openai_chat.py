"""Small OpenAI-compatible transport, separate from rhetorical prompts."""
from __future__ import annotations
import os


class OpenAIChat:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0,
                 top_p: float = 1, max_tokens: int = 1024,
                 disable_thinking: bool = False,
                 api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_BASE_URL"):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"Set the {api_key_env} environment variable before an LLM run.")
        self.client = OpenAI(api_key=key, base_url=os.environ.get(base_url_env) or None,
                             timeout=120, max_retries=2)
        self.model, self.temperature, self.top_p = model, temperature, top_p
        self.max_tokens = max_tokens
        # Reasoning models spend max_tokens on hidden thinking before answering.
        # Turning thinking off keeps rewriting latency and cost bounded; it is a
        # deviation from a "plain" chat model, so it stays opt-in and recorded.
        self.disable_thinking = disable_thinking
        self.usage = {"completion_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, system: str, user: str) -> str:
        extra = {"extra_body": {"thinking": {"type": "disabled"}}} if self.disable_thinking else {}
        response = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, top_p=self.top_p,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        self.usage["completion_calls"] += 1
        if response.usage is not None:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens
            self.usage["completion_tokens"] += response.usage.completion_tokens
        if not response.choices:
            raise RuntimeError("LLM returned no choices.")
        choice = response.choices[0]
        text = choice.message.content
        if choice.finish_reason != "stop" or not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"LLM did not return a complete text response (finish_reason={choice.finish_reason}).")
        return text.strip()
