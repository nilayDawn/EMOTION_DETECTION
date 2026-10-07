from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Input text sentence to classify emotion for",
        examples=["I woke up feeling incredibly optimistic about what the future has in store for me."]
    )


class EmotionScore(BaseModel):
    emotion: str
    emoji: str
    probability: float


class PredictResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    probabilities: Dict[str, float]
    breakdown: List[EmotionScore]


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of text sentences to analyze",
        examples=[[
            "I woke up feeling incredibly optimistic about what the future has in store for me.",
            "I was absolutely livid when I found out they had lied to me for months."
        ]]
    )


class BatchPredictResponse(BaseModel):
    total_count: int
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tokenizer_loaded: bool
    model_path: Optional[str] = None
    classes: List[str]
    emojis: Dict[str, str]
