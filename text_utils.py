import easyocr
import matplotlib.pyplot as plt
import cv2

class TextDetector:
    def __init__(self, GPU_enabled=False):
        self.Reader = easyocr.Reader(['en'], gpu=GPU_enabled)

    def DetectAndDisplay(self, image, threshold=0.2):
        text_info = self.Reader.readtext(image)
        for text in text_info:

            bounding_box, text, score = text

            if score > threshold:
                # We can use the bounding box values to paint rectangles using OpenCV
                # Turn the coordinates in floats tot tuples of ints
                pt1 = tuple(map(int, bounding_box[0]))
                pt2 = tuple(map(int, bounding_box[2]))
                cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
                text_origin = tuple(map(int, bounding_box[0]))
                cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

        return image

    def DetectAndReplace(self, image, threshold=0.2):
        text_info = self.Reader.readtext(image)
        for text in text_info:

            bounding_box, text, score = text

            if score > threshold:
                # We can use the bounding box values to paint rectangles using OpenCV
                # Turn the coordinates in floats tot tuples of ints
                pt1 = tuple(map(int, bounding_box[0]))
                pt2 = tuple(map(int, bounding_box[2]))
                cv2.rectangle(image, pt1, pt2, (255, 255, 255), -1)

                # Compute center of the bounding box
                x_coords = [point[0] for point in bounding_box]
                y_coords = [point[1] for point in bounding_box]
                cx = int(sum(x_coords) / 4)
                cy = int(sum(y_coords) / 4)

                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)
                text_x = cx - text_width // 2
                text_y = cy + text_height // 2

                cv2.putText(image, text, (text_x,text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

        return image