from sqlalchemy import Column, Integer, String
from db import Base

class Media(Base):
    __tablename__ = "media"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    number_speaker = Column(Integer, nullable=True)
    audio_path = Column(String, nullable=True)
    video_path = Column(String, nullable=True)
