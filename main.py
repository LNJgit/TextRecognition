import cv2
import easyocr
import matplotlib.pyplot as plt
import mediapipe as mp
from face_utils import FaceDetection
from text_utils import TextDetector


#Read image
image_path = 'assets/test1.jpg'

img = cv2.imread(image_path)

#Instantiate our face detection object
FaceDetection = FaceDetection()
img = FaceDetection.blurFace(img)

#Instantiate our text detection object
TextDetector = TextDetector(GPU_enabled = True)
img = TextDetector.DetectAndReplace(img)


#Show image with matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()