import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "C:\Users\acer\Downloads\projectt\sign\converted_keras\keras_model.h5",  # Specify your model file
    "C:\Users\acer\Downloads\projectt\sign\converted_keras\labels.txt"  # Specify your labels file if needed
)

# Constants
offset = 20
imgSize = 300
labels = ["Hello", "Thank you"]

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]), 
                       max(0, x - offset):min(x + w + offset, img.shape[1])]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgWhite[:, (imgSize - wCal) // 2:(imgSize + wCal) // 2] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgWhite[(imgSize - hCal) // 2:(imgSize + hCal) // 2, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Draw bounding box and label
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), 
                      (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)  
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2) 
        cv2.rectangle(imgOutput, (x - offset, y - offset), 
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)   

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow exiting by pressing 'q'
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
