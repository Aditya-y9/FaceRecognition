import cv2
# os to create labels for our training data
# to handle file related operatons
import os
# numpy to convert python list to numpy array as 
# open cv face recognizer accepts numpy array
import numpy as np

def faceDetection(test_img):
    # convert color image to grayscale
    # as opencv face detector accept gray images

    # using opencv cvtColor function
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # load haar classifier
    # haar classifier is a machine learning based approach
    # where a cascade function is trained from a lot of positive and negative images
    # to detect faces
    face_haar_cascade = cv2.CascadeClassifier(r"C:\Users\MSHOME\Desktop\Newfolder\FaceRecognition\HaarCascade\haarcascade_frontalface_default.xml")
    # detect multiscale images
    # classifier is loaded and image is passed to detectMultiScale function

    # returns the rectangle values of detected faces
    # rectangle values are stored in faces
    # scale factor specifies how much the image size is reduced with each scale
    # minNeighbors specifies how many neighbors each candidate rectangle should have
    # to retain it
    # images bigger in size are likely to be not detected
    # so we reduce the size by 1.32 times
    # minNeighbors = 5 means that a rectangle should have 5 neighbors to be called a face

    # returns a list of rectangles
    # rectangles are stored in faces

    # to prevent false positives

    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)
    return faces, gray_img


