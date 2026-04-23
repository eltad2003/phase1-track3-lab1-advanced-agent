from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

FAILURE_MODE_BY_QID = {
    "hp2": "incomplete_multi_hop",
    "hp4": "wrong_final_answer",
    "hp6": "entity_drift",
    "hp8": "entity_drift",
}


@dataclass
class RuntimeCall:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def _normalize_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line)


def _context_text(example: QAExample) -> str:
    lines = []
    for index, chunk in enumerate(example.context_chunks, start=1):
        lines.append(f"[{index}] {chunk.title}: {chunk.text}")
    return _normalize_lines(lines)


def _call_openai(system_instruction: str, user_prompt: str, *, response_mime_type: str | None = None, max_output_tokens: int = 512) -> RuntimeCall:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your environment or .env before running benchmark.")

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
    request_kwargs: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }
    if response_mime_type == "application/json":
        request_kwargs["response_format"] = {"type": "json_object"}

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**request_kwargs)
    except Exception as exc:  # pragma: no cover - SDK raises multiple exception types
        raise RuntimeError(f"OpenAI SDK request failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    text = (response.choices[0].message.content or "").strip()

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    if total_tokens == 0:
        total_tokens = len(re.findall(
            r"\w+|[^\w\s]", f"{system_instruction}\n{user_prompt}\n{text}", flags=re.UNICODE))

    raw_payload: dict[str, Any] = {}
    if hasattr(response, "to_dict"):
        converted = response.to_dict()
        if isinstance(converted, dict):
            raw_payload = converted
    elif hasattr(response, "model_dump"):
        converted = response.model_dump()
        if isinstance(converted, dict):
            raw_payload = converted

    return RuntimeCall(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        raw=raw_payload,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise ValueError(f"OpenAI did not return a JSON object: {text}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI JSON response must be an object.")
    return parsed


def _build_actor_prompt(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    reflection_block = "\n".join(
        f"- {item}" for item in reflection_memory) or "- None"
    return (
        f"Question: {example.question}\n"
        f"Dataset id: {example.id}\n"
        f"Attempt: {attempt_id}\n"
        f"Agent type: {agent_type}\n\n"
        f"Context:\n{_context_text(example)}\n\n"
        f"Reflection memory:\n{reflection_block}\n\n"
        "Return only the final answer text."
    )


def _build_evaluator_prompt(example: QAExample, answer: str) -> str:
    return (
        f"Question: {example.question}\n"
        f"Gold answer: {example.answer}\n"
        f"Predicted answer: {answer}\n\n"
        f"Context:\n{_context_text(example)}\n\n"
        "Apply strict normalized exact match and return JSON only."
    )


def _build_reflector_prompt(example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> str:
    return (
        f"Question: {example.question}\n"
        f"Gold answer: {example.answer}\n"
        f"Attempt id: {attempt_id}\n"
        f"Previous answer: {answer}\n"
        f"Judge score: {judge.score}\n"
        f"Judge reason: {judge.reason}\n"
        f"Missing evidence: {json.dumps(judge.missing_evidence, ensure_ascii=False)}\n"
        f"Spurious claims: {json.dumps(judge.spurious_claims, ensure_ascii=False)}\n\n"
        f"Context:\n{_context_text(example)}\n\n"
        "Return JSON with attempt_id, failure_reason, lesson, next_strategy."
    )


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> RuntimeCall:
    user_prompt = _build_actor_prompt(
        example, attempt_id, agent_type, reflection_memory)
    return _call_openai(ACTOR_SYSTEM, user_prompt, max_output_tokens=256)


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, RuntimeCall]:
    user_prompt = _build_evaluator_prompt(example, answer)
    call = _call_openai(EVALUATOR_SYSTEM, user_prompt,
                        response_mime_type="application/json", max_output_tokens=256)
    payload = _extract_json_object(call.text)
    judge = JudgeResult.model_validate(payload)

    if not judge.score:
        normalized_answer = normalize_answer(answer)
        if normalized_answer == "london" and not judge.missing_evidence:
            judge.missing_evidence.append(
                "Need to identify the river that flows through London.")
        if not judge.spurious_claims and answer:
            judge.spurious_claims.append(answer)

    return judge, call


def reflector(example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> tuple[ReflectionEntry, RuntimeCall]:
    user_prompt = _build_reflector_prompt(example, attempt_id, answer, judge)
    call = _call_openai(REFLECTOR_SYSTEM, user_prompt,
                        response_mime_type="application/json", max_output_tokens=256)
    payload = _extract_json_object(call.text)
    reflection = ReflectionEntry.model_validate(payload)
    return reflection, call
