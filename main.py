import cv2
import easyocr
import matplotlib.pyplot as plt

#Read image
image_path = 'assets/test1.jpg'

img = cv2.imread(image_path)

#Instantiate the reader
reader = easyocr.Reader(['en'], gpu=True)

#Detect text with reader
text = reader.readtext(img)

#From our data we get bounding box, text and score values
for t in text:
    bounding_box, text, score = t

    #We can use the bounding box values to paint rectangles using OpenCV
    cv2.rectangle(img, bounding_box[0], bounding_box[2], (255, 0, 0), 2)
    cv2.putText(img,text,bounding_box[0],cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)


#Show image with matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()