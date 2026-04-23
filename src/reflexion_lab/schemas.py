from __future__ import annotations
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    title: str
    text: str


class SupportingFacts(BaseModel):
    title: list[str]
    sent_id: list[int]


class HotpotContext(BaseModel):
    title: list[str]
    sentences: list[list[str]]


class QAExample(BaseModel):
    id: str
    question: str
    answer: str
    type: str
    level: Literal["easy", "medium", "hard"]
    supporting_facts: SupportingFacts
    context: HotpotContext

    @property
    def qid(self) -> str:
        return self.id

    @property
    def gold_answer(self) -> str:
        return self.answer

    @property
    def context_chunks(self) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []
        for title, sentences in zip(self.context.title, self.context.sentences):
            text = " ".join(sentence.strip() for sentence in sentences).strip()
            chunks.append(ContextChunk(title=title, text=text))
        return chunks


class JudgeResult(BaseModel):
    # DONE: Học viên định nghĩa các trường cần thiết cho kết quả đánh giá (score, reason, ...)
    score: int
    reason: str
    missing_evidence: list[str] = Field(default_factory=list)
    spurious_claims: list[str] = Field(default_factory=list)


class ReflectionEntry(BaseModel):
    # DONE: Học viên định nghĩa các trường cần thiết cho một mục reflection (attempt_id, lesson, strategy, ...)
    attempt_id: int
    failure_reason: str
    lesson: str
    next_strategy: str


class AttemptTrace(BaseModel):
    attempt_id: int
    answer: str
    score: int
    reason: str
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = 0
    latency_ms: int = 0


class RunRecord(BaseModel):
    qid: str
    question: str
    gold_answer: str
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str
    is_correct: bool
    attempts: int
    token_estimate: int
    latency_ms: int
    failure_mode: Literal["none", "entity_drift", "incomplete_multi_hop",
                          "wrong_final_answer", "looping", "reflection_overfit"]
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)


class ReportPayload(BaseModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str


class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
