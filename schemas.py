from pydantic import BaseModel
from typing import Optional

class MediaBase(BaseModel):
    name: str
    description: Optional[str] = None
    number_speaker: Optional[int] = None

class MediaResponse(MediaBase):
    id: int
    audio_path: Optional[str] = None
    video_path: Optional[str] = None

    class Config:
        orm_mode = True
