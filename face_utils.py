from typing import List, Tuple, Optional
import numpy as np
import cv2
import mediapipe as mp

Box = Tuple[int, int, int, int]

class FaceDetection:
    def __init__(self,
                 model_selection: int = 0,
                 min_detection_confidence: float = 0.5) -> None:
        self._mp_fd = mp.solutions.face_detection
        self.detector = self._mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        self.model_selection = model_selection
        self.min_conf = min_detection_confidence

    def close(self) -> None:
        if getattr(self, "detector", None) is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _rel_to_abs_bbox(rel_box, W: int, H: int) -> Box:
        x = int(rel_box.xmin * W)
        y = int(rel_box.ymin * H)
        w = int(rel_box.width * W)
        h = int(rel_box.height * H)
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(0, min(w, W - x))
        h = max(0, min(h, H - y))
        return x, y, w, h

    def detect(self, img_bgr: np.ndarray) -> Tuple[List[Box], List]:
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.process(img_rgb)
        detections = result.detections or []
        boxes: List[Box] = []
        for det in detections:
            rel = det.location_data.relative_bounding_box
            boxes.append(self._rel_to_abs_bbox(rel, W, H))
        return boxes, detections

    @staticmethod
    def draw_boxes(img_bgr: np.ndarray,
                   boxes: List[Box],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        out = img_bgr.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out

    def detect_and_draw(self,
                        img_bgr: np.ndarray,
                        color: Tuple[int, int, int] = (0, 0, 0),
                        thickness: int = 1) -> Tuple[np.ndarray, List]:
        boxes, detections = self.detect(img_bgr)
        out = self.draw_boxes(img_bgr, boxes, color=color, thickness=thickness)
        return out, detections

    def blur_faces(self,
                   img_bgr: np.ndarray,
                   kernel: Tuple[int, int] = (50, 50),
                   draw_border: bool = True,
                   border_color: Tuple[int, int, int] = (0, 0, 0),
                   border_thickness: int = 1) -> Tuple[np.ndarray, List[Box]]:
        out = img_bgr.copy()
        boxes, _ = self.detect(out)
        for (x, y, w, h) in boxes:
            if w <= 0 or h <= 0:
                continue
            roi = out[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            out[y:y + h, x:x + w] = cv2.blur(roi, kernel)
            if draw_border:
                cv2.rectangle(out, (x, y), (x + w, y + h), border_color, border_thickness)
        return out, boxes
