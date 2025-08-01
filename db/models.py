from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

#Defines the data class for images
class Image(Base):
    __tablename__ = 'ocr_images'

    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    date_added = Column(DateTime, default=datetime.utcnow)

    words = relationship("Word", back_populates="image")

#Defines the data class for words
class Word(Base):
    __tablename__ = 'ocr_words'

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    cx = Column(Integer)
    cy = Column(Integer)
    image_id = Column(Integer, ForeignKey('ocr_images.id'))

    image = relationship("Image", back_populates="words")

