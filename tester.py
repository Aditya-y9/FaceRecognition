import cv2
import os
import numpy as np
import main as m

# this function will read all persons' training images, detect face from each image
# used to load an image from a file
test_img = cv2.imread(r"C:\Users\MSHOME\Desktop\Newfolder\FaceRecognition\Images\Aditya.test.jpg")

# detect faces from test image

# collect the rectangles returned by faceDetection function
# collect the gray image returned by faceDetection function
faces_detected, gray_img = m.faceDetection(test_img)

print("faces_detected:", faces_detected)

for(x,y,w,h) in faces_detected:
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,102,0), thickness=2, lineType=2, shift=0)

# resized_img = cv2.resize(test_img, (1000,700))
cv2.imshow("face detection tutorial", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
