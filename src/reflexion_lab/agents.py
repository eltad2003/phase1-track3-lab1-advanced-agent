from __future__ import annotations
from dataclasses import dataclass
import re
import time
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


def _count_runtime_tokens(text: str) -> int:
    """Count pseudo-tokens from actual runtime text (words and punctuation)."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _join_context(example: QAExample) -> str:
    return "\n".join(f"{chunk.title}: {chunk.text}" for chunk in example.context_chunks)


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        context_text = _join_context(example)
        for attempt_id in range(1, self.max_attempts + 1):
            attempt_start = time.perf_counter()
            answer = actor_answer(example, attempt_id,
                                  self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            # DONE: Replace with actual token count from runtime I/O text
            token_text = "\n".join(
                [
                    example.question,
                    context_text,
                    "\n".join(reflection_memory),
                    answer,
                    judge.reason,
                    "\n".join(judge.missing_evidence),
                    "\n".join(judge.spurious_claims),
                ]
            )
            token_estimate = _count_runtime_tokens(token_text)
            # DONE: Replace with actual latency measurement
            latency_ms = int((time.perf_counter() - attempt_start) * 1000)
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score,
                                 reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break

            # DONE: Học viên triển khai logic Reflexion tại đây
            # 1. Kiểm tra nếu agent_type là 'reflexion' và chưa hết số lần attempt
            # 2. Gọi hàm reflector để lấy nội dung reflection
            # 3. Cập nhật reflection_memory để Actor dùng cho lần sau
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflection_memory.append(reflection.next_strategy)
                reflections.append(reflection)
                trace.reflection = reflection
                token_estimate += _count_runtime_tokens(
                    "\n".join([reflection.failure_reason,
                              reflection.lesson, reflection.next_strategy])
                )
                trace.token_estimate = token_estimate

            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(
            example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
