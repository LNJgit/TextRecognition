import re
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import easyocr
from spellchecker import SpellChecker

Word = Dict[str, object]
Line = List[Word]

def clean_ocr_line(line: str, spell: Optional[SpellChecker] = None) -> str:
    spell = spell or SpellChecker()
    words = line.split()

    corrected: List[str] = []
    for w in words:
        low = w.lower()
        if len(low) <= 2 or any(ch.isdigit() for ch in low) or (w.isupper() and len(w) <= 4):
            corrected.append(low)
            continue
        cand = spell.correction(low)
        corrected.append(cand if cand is not None else low)

    if corrected:
        corrected[0] = corrected[0].capitalize()
    return ' '.join(corrected)

def _normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip())

def group_and_sort_text_lines(text_info, y_overlap_threshold: float = 0.5) -> List[Line]:
    processed: List[Word] = []
    for bbox, text, score in text_info:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        cx = sum(x_coords) / 4
        cy = sum(y_coords) / 4
        height = max(y_coords) - min(y_coords)
        processed.append({'cy': cy, 'cx': cx, 'text': text, 'bbox': bbox, 'score': score, 'height': height})

    lines: List[Line] = []
    for word in processed:
        placed = False
        for line in lines:
            line_cy = sum(w['cy'] for w in line) / len(line)
            line_height = sum(w['height'] for w in line) / len(line)
            if abs(word['cy'] - line_cy) < line_height * y_overlap_threshold:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    lines.sort(key=lambda line: min(w['cy'] for w in line))
    for line in lines:
        line.sort(key=lambda w: w['cx'])
    return lines

def _union_bbox(line: Line, border: int, W: int, H: int) -> Tuple[int, int, int, int]:
    xs = [p[0] for w in line for p in w['bbox']]  # type: ignore
    ys = [p[1] for w in line for p in w['bbox']]  # type: ignore
    x1 = max(int(min(xs)) - border, 0)
    y1 = max(int(min(ys)) - border, 0)
    x2 = min(int(max(xs)) + border, W)
    y2 = min(int(max(ys)) + border, H)
    return x1, y1, x2, y2

def _line_center(line: Line) -> Tuple[int, int]:
    x_coords = [pt[0] for w in line for pt in w['bbox']]  # type: ignore
    y_coords = [pt[1] for w in line for pt in w['bbox']]  # type: ignore
    cx = int(sum(x_coords) / len(x_coords))
    cy = int(sum(y_coords) / len(y_coords))
    return cx, cy

def _sample_background(image: np.ndarray, line: Line, border: int) -> Tuple[int, int, int]:
    H, W = image.shape[:2]
    x1, y1, x2, y2 = _union_bbox(line, border, W, H)

    mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
    for w in line:
        b = w['bbox']  # type: ignore
        p1 = tuple(map(int, b[0])); p2 = tuple(map(int, b[2]))
        px1 = max(p1[0] - x1, 0); py1 = max(p1[1] - y1, 0)
        px2 = min(p2[0] - x1, x2 - x1); py2 = min(p2[1] - y1, y2 - y1)
        mask[py1:py2, px1:px2] = 0

    roi = image[y1:y2, x1:x2]
    mean_color = cv2.mean(roi, mask=mask)[:3]
    return tuple(map(int, mean_color))

def _flatten_subtokens(lines: List[Line]) -> Tuple[List[str], List[int]]:
    subtokens: List[str] = []
    counts: List[int] = []
    for line in lines:
        for w in line:
            parts = _normalize_text(w['text']).split()  # type: ignore
            counts.append(len(parts))
            subtokens.extend(parts)
    return subtokens, counts

def _reassign_subtokens(lines: List[Line], corrected_subtokens: List[str], counts: List[int]) -> None:
    it = iter(corrected_subtokens)
    idx = 0
    for line in lines:
        for w in line:
            k = counts[idx]
            w['text'] = ' '.join(next(it) for _ in range(k)) if k > 0 else ''  # type: ignore
            idx += 1

class TextDetector:
    def __init__(self, GPU_enabled: bool = False, languages: Optional[List[str]] = None):
        self.Reader = easyocr.Reader(languages or ['en'], gpu=GPU_enabled)
        self.spell = SpellChecker()

    def DetectAndDisplay(self, image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        raw_text_info = self.Reader.readtext(image)
        lines = group_and_sort_text_lines(raw_text_info)
        for line in lines:
            for word in line:
                if word['score'] > threshold:  # type: ignore
                    pt1 = tuple(map(int, word['bbox'][0]))  # type: ignore
                    pt2 = tuple(map(int, word['bbox'][2]))  # type: ignore
                    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
                    cv2.putText(image, word['text'], pt1, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)  # type: ignore
        return image

    def DetectAndReplace(
        self,
        image: np.ndarray,
        threshold: float = 0.2,
        border: int = 3,
        font=cv2.FONT_HERSHEY_DUPLEX,
        font_scale: float = 1.0,
        thickness: int = 2,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        debug: bool = False,
    ) -> np.ndarray:

        raw_text_info = self.Reader.readtext(image)
        lines = group_and_sort_text_lines(raw_text_info)

        original_subtokens, subcounts = _flatten_subtokens(lines)
        if debug:
            print(f"[DEBUG] bboxes={sum(1 for _ in (w for ln in lines for w in ln))} "
                  f"subtokens={len(original_subtokens)}")

        cleaned = clean_ocr_line(' '.join(original_subtokens), spell=self.spell)
        corrected_subtokens = cleaned.split()

        if len(corrected_subtokens) != len(original_subtokens):
            raise ValueError(
                f"Subtoken count mismatch: before={len(original_subtokens)} after={len(corrected_subtokens)}"
            )

        _reassign_subtokens(lines, corrected_subtokens, subcounts)

        H, W = image.shape[:2]
        for line in lines:
            x1, y1, x2, y2 = _union_bbox(line, border, W, H)
            cx, cy = _line_center(line)

            bg = _sample_background(image, line, border)
            for w in line:
                p1 = tuple(map(int, w['bbox'][0]))  # type: ignore
                p2 = tuple(map(int, w['bbox'][2]))  # type: ignore
                cv2.rectangle(image, p1, p2, bg, -1)

            line_text = ' '.join(w['text'] for w in line)  # type: ignore
            (tw, th), _ = cv2.getTextSize(line_text, font, font_scale, thickness)
            text_x = max(cx - tw // 2, 0)
            text_y = max(min(cy + th // 2, H - 1), 0)
            cv2.putText(image, line_text, (text_x, text_y), font, font_scale, text_color, thickness)

        return image
