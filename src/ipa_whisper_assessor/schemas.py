from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IpaWord(BaseModel):
    ipa: str
    start: Optional[float] = None
    end: Optional[float] = None
    raw: Optional[str] = None


class TranscriptionResult(BaseModel):
    audio_path: str
    model: str
    ipa_text: str
    ipa_words: list[IpaWord] = Field(default_factory=list)


class PhonemeOp(BaseModel):
    op: Literal["match", "sub", "ins", "del"]
    expected: Optional[str] = None
    predicted: Optional[str] = None


class WordAlignment(BaseModel):
    word_index: int
    reference_word: str
    expected_ipa: str
    predicted_ipa: str
    start: Optional[float] = None
    end: Optional[float] = None
    phoneme_ops: list[PhonemeOp] = Field(default_factory=list)


class MistakeEvent(BaseModel):
    rule: str
    expected: Optional[str] = None
    predicted: Optional[str] = None
    count: int = 1


class AssessmentMetrics(BaseModel):
    phoneme_error_rate: float
    substitutions: int
    insertions: int
    deletions: int


class AssessmentResult(BaseModel):
    audio_path: str
    reference: str
    g2p_backend: str
    model: str
    transcription: TranscriptionResult
    word_alignments: list[WordAlignment]
    metrics: AssessmentMetrics
    substitution_histogram: dict[str, int] = Field(default_factory=dict)
    mistakes: list[MistakeEvent] = Field(default_factory=list)

