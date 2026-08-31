"""Shared low-level model-calling client used by every council role
(proposal generator, critics, chairman, judge, strategy memory). Talks to
Ollama's native /api/chat or an OpenAI-compatible /v1/chat/completions
endpoint (auto-detected), off the asyncio event loop via a thread pool."""

import asyncio
import logging
import time

import requests

from council_attack.utils import redact_secrets as redact
from council_attack.utils import strip_thinking

logger = logging.getLogger(__name__)


class ModelClientError(RuntimeError):
    """Raised when a model call fails after retries with a transport/HTTP error,
    distinct from a model that legitimately returned an empty completion. Lets the
    judge path treat an outage as UNSCORED instead of scoring it as harm 0."""


class ModelClient:
    """Resolves per-role endpoints and issues chat-completion calls.

    ``chairman_model``/``judge_model`` are used only to route a call to
    ``chairman_endpoint``/``judge_endpoint`` when the caller doesn't pass an
    explicit ``endpoint=`` override - every other model falls back to
    ``ollama_endpoint`` (or a ``model_endpoints`` override keyed by bare
    model name).
    """

    def __init__(
        self,
        ollama_endpoint: str,
        chairman_endpoint: str,
        judge_endpoint: str,
        chairman_model: str,
        judge_model: str,
        model_endpoints: dict[str, str],
        ollama_num_ctx: int,
        council_think: bool,
        api_key: str | None = None,
        openai_append_v1: bool = True,
        seed: int | None = None,
    ) -> None:
        self.ollama_endpoint = ollama_endpoint
        self.chairman_endpoint = chairman_endpoint
        self.judge_endpoint = judge_endpoint
        self.chairman_model = chairman_model
        self.judge_model = judge_model
        self.model_endpoints = model_endpoints
        self.ollama_num_ctx = ollama_num_ctx
        self.council_think = council_think
        # Optional bearer token for OpenAI-compatible endpoints (OpenAI API,
        # GitHub Models, vLLM behind auth). None -> no Authorization header,
        # preserving the original unauthenticated local-vLLM behaviour.
        self.api_key = api_key
        # Whether to append "/v1" when the OpenAI-compatible base URL doesn't
        # already contain it. True (default) matches the common vLLM
        # behaviour. Set False for endpoints that expose
        # /chat/completions without a /v1 segment (e.g. GitHub Models).
        self.openai_append_v1 = openai_append_v1
        # Optional decoding seed for reproducibility. Threaded into the Ollama
        # `options.seed` and the OpenAI `seed` payload field when not None. Servers
        # honour it best-effort; combine with a fixed run count (see README).
        self.seed = seed

    async def query(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float = 0.7,
        think: bool | None = None,
        endpoint: str | None = None,
        seed: int | None = None,
        raise_on_error: bool = False,
    ) -> str:
        """
        Call an Ollama model via the native /api/chat endpoint in a thread pool.

        Using requests (sync) wrapped with asyncio.to_thread - async wrapper only;
        parallelism only if callers are awaited concurrently (council calls are actually
        sequential to avoid saturating Ollama). We bypass litellm here because litellm.acompletion returns
        No timeout is set - requests wait as long as the model needs.
        """
        # Resolve target endpoint
        if model == self.judge_model:
            default_ep = self.judge_endpoint
        elif model == self.chairman_model:
            default_ep = self.chairman_endpoint
        else:
            default_ep = endpoint or self.ollama_endpoint

        # Check routing table for bare model name (remove providers)
        bare_model = model.removeprefix("ollama/").removeprefix("openai/")
        target_endpoint = self.model_endpoints.get(bare_model, default_ep)

        model_name = model.removeprefix("ollama/")

        # Detect if OpenAI/vLLM endpoint vs Ollama
        # OpenAI-compatible servers (e.g. vLLM) usually expose a /v1 path
        is_openai = "/v1" in target_endpoint or ":8000" in target_endpoint or model.startswith("openai/")

        effective_think = think if think is not None else self.council_think
        effective_seed = seed if seed is not None else self.seed
        # Some Ollama models (qwen2.5, llama3.x, mistral*, gemma*, mixtral)
        # reject the ``think`` parameter with HTTP 400. Auto-detect: only send
        # ``think`` for known thinking-capable families (qwen3, deepseek-r1).
        supports_think = self._supports_thinking(model_name)
        if effective_think and not supports_think:
            logger.debug(
                "%s does not support thinking; dropping think param", model_name
            )

        def _call() -> str:
            headers = None
            if is_openai:
                # vLLM / OpenAI / GitHub Models format
                base_url = target_endpoint.rstrip('/')
                if base_url.endswith("/chat/completions"):
                    # Base already points at the method.
                    full_url = base_url
                elif "/v1" in base_url:
                    # Already versioned (e.g. vLLM at .../v1): just append method.
                    full_url = f"{base_url}/chat/completions"
                elif self.openai_append_v1:
                    # Default OpenAI/vLLM shape: insert the /v1 segment.
                    full_url = f"{base_url}/v1/chat/completions"
                else:
                    # Endpoints exposing /chat/completions without a /v1 segment
                    # (e.g. GitHub Models: https://models.github.ai/inference).
                    full_url = f"{base_url}/chat/completions"

                if self.api_key:
                    headers = {"Authorization": f"Bearer {self.api_key}"}

                # OpenAI payload structure
                # vLLM on this server expects the full name "openai/gpt-oss-120b"
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
                if effective_seed is not None:
                    payload["seed"] = effective_seed
            else:
                # Ollama format
                chat_path = "/api/chat"
                full_url = f"{target_endpoint.rstrip('/')}{chat_path}"
                # This Ollama branch only serves council members + chairman, all of
                # which we want pinned in VRAM (keep_alive: -1) to avoid reload
                # thrashing across turns. num_ctx must be set explicitly or Ollama
                # falls back to its small 2048 default and truncates long histories.
                payload: dict = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": self.ollama_num_ctx,
                    },
                }
                if effective_seed is not None:
                    payload["options"]["seed"] = effective_seed
                if supports_think:
                    payload["think"] = effective_think

            # Hard cap: 30 minutes per call. A `timeout=None` can hang
            # indefinitely if the backend crashes at the judge endpoint,
            # making the council wait forever instead of erroring out.
            # 1800s is more than enough for any single generation;
            # transient errors raise and bubble up so an outer retry can
            # take over.
            last_exc: Exception | None = None
            for _attempt in range(3):
                try:
                    resp = requests.post(
                        full_url,
                        json=payload,
                        headers=headers,
                        timeout=1800,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if is_openai:
                        # OpenAI response format
                        choices = data.get("choices", [])
                        if not choices:
                            return ""
                        return choices[0].get("message", {}).get("content", "")
                    else:
                        # Ollama response format
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        thinking = msg.get("thinking", "")
                        return content if content else thinking
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Model %s failed at %s (attempt %d/3): %s",
                        model_name, redact(full_url), _attempt + 1, redact(str(exc)),
                    )
                    time.sleep(min(2 ** _attempt, 5))

            # All retries exhausted: a persistent transport/HTTP error, NOT an empty
            # completion. When raise_on_error is set (the judge/target path), raise so
            # the caller marks the turn UNSCORED rather than scoring the outage as
            # harm 0; otherwise degrade to "" as before (surrogate roles tolerate it).
            if raise_on_error:
                raise ModelClientError(
                    f"{model_name} failed after retries: {redact(str(last_exc))}"
                )
            return ""

        raw = await asyncio.to_thread(_call)
        # Strip reasoning unconditionally: strip_thinking removes <think> blocks AND the
        # GPT-OSS Harmony final-channel wrapper, and is a safe superset of .strip(), so
        # the critique/judge/proposal parsers never see un-stripped reasoning.
        return strip_thinking(raw)

    @staticmethod
    def _supports_thinking(model_name: str) -> bool:
        """Return True only for Ollama model families that accept the ``think``
        parameter on /api/chat (qwen3, deepseek-r1). Everything else (qwen2.5,
        llama3.x, mistral*, gemma*, mixtral, gpt-oss) is rejected with HTTP 400 by
        Ollama, so we omit the param entirely for those."""
        name = model_name.lower()
        if name.startswith("qwen3"):
            return True
        if name.startswith("deepseek-r1"):
            return True
        return False
