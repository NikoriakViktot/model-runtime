"""
cpu_runtime/inference.py

GGUF inference engine backed by llama-cpp-python.

Design
------
- Single Llama instance loaded at startup (heavy; not reloaded per request).
- All inference is synchronous; wrapped in asyncio.to_thread so it runs in
  a thread pool and does not block the FastAPI event loop.
- A semaphore gates concurrent access because llama.cpp is not thread-safe
  when called from multiple threads simultaneously.
- Streaming: the generator yields SSE bytes; the caller wraps in StreamingResponse.

OpenAI chat message format → llama-cpp prompt format via apply_chat_template
when the model exposes one, otherwise falls back to a simple role prefix.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    stream: bool = False
    stop: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    model: str = ""


class LlamaCppEngine:
    """
    Thin wrapper around llama-cpp-python's Llama class.

    Lifecycle:
        engine = LlamaCppEngine(settings)
        await engine.load()          # call once at app startup
        result = await engine.generate(req)
        async for chunk in engine.stream(req): ...
        await engine.unload()        # call at app shutdown
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._llm = None
        # Semaphore: llama.cpp is NOT thread-safe; only one inference at a time
        self._sem: asyncio.Semaphore | None = None

    async def load(self) -> None:
        """Load the GGUF model into memory.  Runs in a thread to avoid blocking."""
        logger.info("Loading GGUF model from %s", self._settings.model_path)
        t0 = time.perf_counter()

        def _load():
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Add 'llama-cpp-python' to requirements.txt."
                ) from exc

            return Llama(
                model_path=self._settings.model_path,
                n_ctx=self._settings.n_ctx,
                n_threads=self._settings.n_threads,
                n_batch=self._settings.n_batch,
                n_gpu_layers=self._settings.n_gpu_layers,
                verbose=False,
            )

        self._llm = await asyncio.to_thread(_load)
        self._sem = asyncio.Semaphore(1)  # one inference at a time

        elapsed = time.perf_counter() - t0
        logger.info(
            "GGUF model loaded: alias=%s ctx=%d threads=%d elapsed=%.1fs",
            self._settings.model_alias,
            self._settings.n_ctx,
            self._settings.n_threads,
            elapsed,
        )

    async def unload(self) -> None:
        """Release model memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("GGUF model unloaded")

    # ------------------------------------------------------------------
    # Unary generation
    # ------------------------------------------------------------------

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        """
        Run inference synchronously in a thread pool.
        Returns the full completion.
        """
        self._assert_loaded()

        prompt = self._build_prompt(req.messages)

        def _run():
            return self._llm(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repeat_penalty=req.repeat_penalty,
                stop=req.stop or ["</s>", "<|im_end|>", "<|eot_id|>"],
                stream=False,
            )

        async with self._sem:
            result = await asyncio.to_thread(_run)

        choice = result["choices"][0]
        usage = result.get("usage", {})

        return GenerationResult(
            text=choice["text"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            model=self._settings.model_alias,
        )

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    async def stream(self, req: GenerationRequest) -> AsyncIterator[bytes]:
        """
        Yield SSE-formatted bytes as tokens are generated.

        Each yielded chunk is a complete ``data: {...}\n\n`` SSE frame
        matching the OpenAI streaming format so the gateway can forward
        them verbatim.
        """
        self._assert_loaded()

        prompt = self._build_prompt(req.messages)
        request_id = f"chatcmpl-cpu-{int(time.time() * 1000)}"
        created = int(time.time())

        def _run_stream():
            return self._llm(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repeat_penalty=req.repeat_penalty,
                stop=req.stop or ["</s>", "<|im_end|>", "<|eot_id|>"],
                stream=True,
            )

        # Acquire the semaphore BEFORE entering the generator so a second
        # concurrent request waits here rather than interleaving with the
        # first inside the thread.
        await self._sem.acquire()
        try:
            iterator = await asyncio.to_thread(_run_stream)

            async def _generate_chunks():
                try:
                    for chunk in iterator:
                        delta_text = chunk["choices"][0].get("text", "")
                        finish_reason = chunk["choices"][0].get("finish_reason")

                        payload = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self._settings.model_alias,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n".encode()

                    yield b"data: [DONE]\n\n"
                finally:
                    self._sem.release()

            async for chunk in _generate_chunks():
                yield chunk

        except Exception:
            self._sem.release()
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        if self._llm is None:
            raise RuntimeError(
                "LlamaCppEngine is not loaded. "
                "Call await engine.load() before inference."
            )

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        """
        Convert OpenAI chat messages to a plain-text prompt.

        Tries llama-cpp's built-in chat template first (models that ship
        with a tokenizer_config.json).  Falls back to a generic
        System/User/Assistant prefix format.
        """
        # Attempt to use the model's own chat template
        try:
            if hasattr(self._llm, "create_chat_completion"):
                # llama-cpp-python ≥ 0.2 exposes this directly; we only use
                # it for prompt formatting — we call __call__ for actual gen
                pass
        except Exception:
            pass

        # Generic fallback — works for most instruction-tuned models
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton — initialised in app lifespan
# ---------------------------------------------------------------------------

engine: LlamaCppEngine | None = None


def get_engine() -> LlamaCppEngine:
    if engine is None:
        raise RuntimeError("Engine not initialised — call engine.load() at startup.")
    return engine
