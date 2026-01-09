from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

BloomLevel = Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
Difficulty = Literal["Easy", "Medium", "Hard"]

class SourceCitation(BaseModel):
    source: str
    page: int
    snippet: str

class QuestionItem(BaseModel):
    id: str = Field(..., description="Unique id like Q1, Q2...")
    question_text: str
    bloom_level: BloomLevel
    co_mapping: str = Field(..., description="Use CO1/CO2/CO3 style tags")
    difficulty: Difficulty
    marks: int = Field(..., ge=1, le=20)
    answer_key: str
    detailed_rubric: str
    source_citation: list[SourceCitation]

class QuestionBank(BaseModel):
    course: str
    questions: list[QuestionItem]


class SubtopicPlan(BaseModel):
    name: str
    importance: int = Field(..., ge=1, le=5)
    why: str
    query: str


class TopicPlan(BaseModel):
    topic: str
    subtopics: list[SubtopicPlan]
    notes: str


class QuestionStyleMix(BaseModel):
    theory: int = Field(..., ge=0, le=100)
    numerical: int = Field(..., ge=0, le=100)
    derivation: int = Field(..., ge=0, le=100)
    equation: int = Field(..., ge=0, le=100)
    diagram: int = Field(..., ge=0, le=100)


class SubjectProfile(BaseModel):
    subject: str
    rationale: str
    recommended_mix: QuestionStyleMix
    common_question_types: list[str]


class AuditIssue(BaseModel):
    id: str | None = None
    category: str
    detail: str


class AuditReport(BaseModel):
    passed: bool
    issues: list[AuditIssue]
    summary: str
