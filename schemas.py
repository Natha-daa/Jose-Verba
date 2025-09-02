# app/schemas.py
from pydantic import BaseModel
from typing import Optional

class MediaCreate(BaseModel):
    name: str
    description: Optional[str] = None
    numberSpeaker: int = 1
    audioLink: Optional[str] = None
    videoLink: Optional[str] = None
    fileSize: Optional[str] = None

class MediaResponse(MediaCreate):
    id: int
    createdAt: Optional[str]

    class Config:
        from_attributes = True
