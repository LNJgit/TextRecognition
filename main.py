import cv2
import easyocr
import matplotlib.pyplot as plt
import mediapipe as mp

#Read image
image_path = 'assets/test3.jpg'

img = cv2.imread(image_path)

#Instantiate the reader
reader = easyocr.Reader(['en'], gpu=True)

#Detect text with reader
text = reader.readtext(img)

#Create a threshold for deciding which texts should be displayed
threshold = 0.2

#Get width and height (Needed for bounding box in face detection)
H,W, _ = img.shape

#Instantiate the face detector
mp_face_detector = mp.solutions.face_detection

with mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if (out.detections is not None):
        for detection in out.detections:
            location_data = detection.location_data
            bounding_box = location_data.relative_bounding_box

            x, y, width, height = bounding_box.xmin, bounding_box.ymin, bounding_box.width,bounding_box.height

            x = int(x*W)
            y = int(y*H)
            width = int(width*W)
            height = int(height*H)

            cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 4)

#From our data we get bounding box, text and score values
for t in text:
    bounding_box, text, score = t

    if score > threshold:
        #We can use the bounding box values to paint rectangles using OpenCV
        #Turn the coordinates in floats tot tuples of ints
        pt1 = tuple(map(int, bounding_box[0]))
        pt2 = tuple(map(int, bounding_box[2]))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
        text_origin = tuple(map(int, bounding_box[0]))
        cv2.putText(img, text, text_origin, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)


#Show image with matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()