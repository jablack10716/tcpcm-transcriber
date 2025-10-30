"""Pydantic schemas for transcription data structures."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A transcription segment with timing and text."""
    id: int
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    """Complete transcript with metadata."""
    segments: List[Segment]
    language: Optional[str] = None
    duration: Optional[float] = None


class Chunk(BaseModel):
    """A text chunk for RAG ingestion with metadata."""
    chunk_id: int
    text: str
    start: float
    end: float
    segment_ids: List[int] = Field(default_factory=list)
    char_count: int
    source_file: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_id": 0,
                "text": "Introduction to TcPCM...",
                "start": 0.0,
                "end": 15.5,
                "segment_ids": [0, 1, 2],
                "char_count": 1200,
                "source_file": "video.mp4"
            }
        }
    }
