import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the hand detector
detector = HandDetector(maxHands=1)

# Parameters for cropping and resizing
offset = 20
imgSize = 300
counter = 0

# Correct the folder path using a raw string
folder = r"C:\Users\acer\Downloads\projectt\sign\Data\Yes"


while True:
    # Read the camera frame
    success, img = cap.read()
    if not success:
        print("Error: Failed to read from webcam.")
        break
    
    # Find hands in the frame
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure the cropping coordinates are within the image dimensions
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region from the frame
        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        # Determine aspect ratio and resize accordingly
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Display the cropped and resized images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the original frame
    cv2.imshow('Image', img)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")

    # To exit the loop cleanly
    if key == 27:  # Press 'Esc' key to exit
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
