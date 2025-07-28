import mediapipe as mp
import cv2

class FaceDetection:
    def __init__(self):
        self.FaceDetector = mp.solutions.face_detection

    def blurFace(self, img):

        H, W, _ = img.shape

        with self.FaceDetector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = face_detection.process(img_rgb)

            if (out.detections is not None):
                for detection in out.detections:
                    location_data = detection.location_data
                    bounding_box = location_data.relative_bounding_box

                    x, y, width, height = bounding_box.xmin, bounding_box.ymin, bounding_box.width, bounding_box.height

                    x = int(x * W)
                    y = int(y * H)
                    width = int(width * W)
                    height = int(height * H)

                    img[y:y + height, x:x + width, :] = cv2.blur(img[y:y + height, x:x + width, :], (50, 50))
                    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 0), 1)

        return img