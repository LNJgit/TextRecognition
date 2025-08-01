from sqlalchemy.orm import Session
from db.models import Image, Word
from typing import List

def save_or_update_ocr_result(
    db: Session,
    image_filename: str,
    words: List[str]
) -> None:
    # Check if image already exists
    img_entry = db.query(Image).filter_by(filename=image_filename).first()

    if img_entry is None:
        img_entry = Image(filename=image_filename)
        db.add(img_entry)
        db.flush()
    else:
        # If exists, delete old words
        db.query(Word).filter_by(image_id=img_entry.id).delete()

    for word_text in words:
        db.add(Word(text=word_text, image_id=img_entry.id))

    db.commit()

def find_images_by_word(db: Session, query: str):
    return (
        db.query(Image)
        .join(Word)
        .filter(Word.text.ilike(f'%{query}%'))
        .all()
    )