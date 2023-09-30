import cv2
import os
import numpy as np
import main as m

# this function will read all persons' training images, detect face from each image
# used to load an image from a file
test_img = cv2.imread(r"C:\Users\MSHOME\Desktop\Newfolder\FaceRecognition\Images\Aditya.jpg")
# vid = cv2.VideoCapture(r"C:\Users\MSHOME\Desktop\Newfolder\FaceRecognition\video\Video.mp4")
# running = True
# while running:
#     success, frame = vid.read()
#     resized_img = cv2.resize(frame, (1000,700))
#     faces_detected, gray_img = m.faceDetection(resized_img)
#     for(x,y,w,h) in faces_detected:
#         cv2.rectangle(resized_img, (x,y), (x+w, y+h), (255,102,0), thickness=2, lineType=8, shift=0)
#     cv2.imshow("face detection tutorial", resized_img)
#     cv2.waitKey(1)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()

# detect faces from test image

# collect the rectangles returned by faceDetection function
# # collect the gray image returned by faceDetection function
faces_detected, gray_img = m.faceDetection(test_img)

# print("faces_detected:", faces_detected)

# for(x,y,w,h) in faces_detected:
#     cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,102,0), thickness=2, lineType=2, shift=0

# resized_img = cv2.resize(test_img, (1000,700))
# cv2.imshow("face detection tutorial", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

faces, faceID = m.labels_for_training_data(r"C:\Users\MSHOME\Desktop\Newfolder\FaceRecognition\TrainingImages")
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read(r"C:\Users\MSHOME\Desktop\Newfolder\FaceRecognition\trainingData.yml")
face_recognizer = m.train_classifier(faces, faceID)
# m.save("trainingData.yml")

name = {0:"Ranbir", 1:"Aditya"}

for faces in faces_detected:
    (x,y,w,h) = faces
    # extracting region of interest
    roi_gray = gray_img[y:y+h, x:x+h]
    # predicting the label of given image
    # confidence is the accuracy of the prediction
    # confidence is a number between 0 and 100
    # the lower the value, the more accurate the prediction
    # label 0 or 1
    # confidence value lower than its more accurate
    # 35 is the threshold value for confidence
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:", label)
    m.draw_rect(test_img, faces)

    # extract the name from the dictionary
    predicted_name = name[label]


    m.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000,700))
cv2.imshow("face detection tutorial", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



