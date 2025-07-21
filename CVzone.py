import cv2
import mediapipe as mp
import time
import math
import numpy as np
import cvzone

# # FaceDetectionBasics.py
#
# # cap = cv2.VideoCapture("Videos/1.mp4")
# cap = cv2.VideoCapture(0)
# pTime = 0
#
# mpFaceDetection = mp.solutions.face_detection
# mpDraw = mp.solutions.drawing_utils
# faceDetection = mpFaceDetection.FaceDetection(0.75)
#
# while True:
#     success, img = cap.read()
#
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceDetection.process(imgRGB)
#     print(results)
#
#     if results.detections:
#         for id, detection in enumerate(results.detections):
#             # mpDraw.draw_detection(img, detection)
#             # print(id, detection)
#             # print(detection.score)
#             # print(detection.location_data.relative_bounding_box)
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, ic = img.shape
#             bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
#                    int(bboxC.width * iw), int(bboxC.height * ih)
#             cv2.rectangle(img, bbox, (255, 0, 255), 2)
#             cv2.putText(img, f'{int(detection.score[0] * 100)}%',
#                         (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
#                         2, (255, 0, 255), 2)
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
#                 3, (0, 255, 0), 2)
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)


# FaceDetectionModule.py

# class FaceDetector():
#     def __init__(self, minDetectionCon=0.5):
#
#         self.minDetectionCon = minDetectionCon
#
#         self.mpFaceDetection = mp.solutions.face_detection
#         self.mpDraw = mp.solutions.drawing_utils
#         self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
#
#     def findFaces(self, img, draw=True):
#
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.faceDetection.process(imgRGB)
#         # print(self.results)
#         bboxs = []
#         if self.results.detections:
#             for id, detection in enumerate(self.results.detections):
#                 bboxC = detection.location_data.relative_bounding_box
#                 ih, iw, ic = img.shape
#                 bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
#                        int(bboxC.width * iw), int(bboxC.height * ih)
#                 bboxs.append([id, bbox, detection.score])
#                 if draw:
#                     img = self.fancyDraw(img,bbox)
#
#                     cv2.putText(img, f'{int(detection.score[0] * 100)}%',
#                             (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
#                             2, (255, 0, 255), 2)
#         return img, bboxs
#
#     def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
#         x, y, w, h = bbox
#         x1, y1 = x + w, y + h
#
#         cv2.rectangle(img, bbox, (255, 0, 255), rt)
#         # Top Left  x,y
#         cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
#         cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
#         # Top Right  x1,y
#         cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
#         cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
#         # Bottom Left  x,y1
#         cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
#         cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
#         # Bottom Right  x1,y1
#         cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
#         cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
#         return img
#
#
# def main():
#     # cap = cv2.VideoCapture("Videos/6.mp4")
#     cap  = cv2.VideoCapture(0)
#     pTime = 0
#     detector = FaceDetector()
#     while True:
#         success, img = cap.read()
#         img, bboxs = detector.findFaces(img)
#         print(bboxs)
#
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#         cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
#
#
# if __name__ == "__main__":
#     main()

# import cv2
# import cvzone  # Importing the cvzone library
#
# # Initialize the webcam
# cap = cv2.VideoCapture(0)  # Capture video from the third webcam (0-based index)
#
# # Main loop to continuously capture frames
# while True:
#     # Capture a single frame from the webcam
#     success, img = cap.read()  # 'success' is a boolean that indicates if the frame was captured successfully, and 'img' contains the frame itself
#
#     # Add a rectangle with styled corners to the image
#     img = cvzone.cornerRect(
#         img,  # The image to draw on
#         (200, 200, 300, 200),  # The position and dimensions of the rectangle (x, y, width, height)
#         l=30,  # Length of the corner edges
#         t=5,  # Thickness of the corner edges
#         rt=1,  # Thickness of the rectangle
#         colorR=(255, 0, 255),  # Color of the rectangle
#         colorC=(0, 255, 0)  # Color of the corner edges
#     )
#
#     # Show the modified image
#     cv2.imshow("Image", img)  # Display the image in a window named "Image"
#
#     # Wait for 1 millisecond between frames
#     cv2.waitKey(1)  # Waits 1 ms for a key event (not being used here)

# Stack Image

# import cv2
# import cvzone
#
# # Initialize camera capture
# cap = cv2.VideoCapture(0)
#
# # Start an infinite loop to continually capture frames
# while True:
#     # Read image frame from camera
#     success, img = cap.read()
#
#     # Convert the image to grayscale
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Resize the image to be smaller (0.1x of original size)
#     imgSmall = cv2.resize(img, (0, 0), None, 0.1, 0.1)
#
#     # Resize the image to be larger (3x of original size)
#     imgBig = cv2.resize(img, (0, 0), None, 3, 3)
#
#     # Apply Canny edge detection on the grayscale image
#     imgCanny = cv2.Canny(imgGray, 50, 150)
#
#     # Convert the image to HSV color space
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # Create a list of all processed images
#     imgList = [img, imgGray, imgCanny, imgSmall, imgBig, imgHSV]
#
#     # Stack the images together using cvzone's stackImages function
#     stackedImg = cvzone.stackImages(imgList, 3, 0.7)
#
#     # Display the stacked images
#     cv2.imshow("stackedImg", stackedImg)
#
#     # Wait for 1 millisecond; this also allows for keyboard inputs
#     cv2.waitKey(1)

# Rotate Image

# import cv2
# from cvzone.Utils import rotateImage  # Import rotateImage function from cvzone.Utils
#
# # Initialize the video capture
# cap = cv2.VideoCapture(0)  # Capture video from the third webcam (index starts at 0)
#
# # Start the loop to continuously get frames from the webcam
# while True:
#     # Read a frame from the webcam
#     success, img = cap.read()  # 'success' will be True if the frame is read successfully, 'img' will contain the frame
#
#     # Rotate the image by 60 degrees without keeping the size
#     imgRotated60 = rotateImage(img, 60, scale=1,
#                                keepSize=False)  # Rotate image 60 degrees, scale it by 1, and don't keep original size
#
#     # Rotate the image by 60 degrees while keeping the size
#     imgRotated60KeepSize = rotateImage(img, 60, scale=1,
#                                        keepSize=True)  # Rotate image 60 degrees, scale it by 1, and keep the original size
#
#     # Display the rotated images
#     cv2.imshow("imgRotated60", imgRotated60)  # Show the 60-degree rotated image without keeping the size
#     cv2.imshow("imgRotated60KeepSize", imgRotated60KeepSize)  # Show the 60-degree rotated image while keeping the size
#
#     # Wait for 1 millisecond between frames
#     cv2.waitKey(1)  # Wait for 1 ms, during which any key press can be detected (not being used here)