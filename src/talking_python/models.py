from typing import TypedDict

from pydantic import BaseModel


class TranscriptSlice(TypedDict):
    filename: str
    contents: list[str]
    lines: list[int]


class TranscriptPassage(BaseModel):
    """A piece from a transcript."""

    id: str
    line: str
    title: str
    embedding: list[float]
