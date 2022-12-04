import cv2
import time
import os
import hand_tracking_module as htm

wCam, hCam = 640,480

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImgs"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fingerCount = 0

    if lmList:
        if lmList[8][2] < lmList[6][2]:
            fingerCount += 1
        if lmList[12][2] < lmList[10][2]:
            fingerCount += 1
        if lmList[16][2] < lmList[14][2]:
            fingerCount += 1
        if lmList[20][2] < lmList[18][2]:
            fingerCount += 1
        if lmList[4][1] > lmList[3][1]:
            fingerCount += 1
    print(fingerCount)
    h, w, c = overlayList[fingerCount].shape
    img[0:h, 0:w] = overlayList[fingerCount]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,0), 3)

    cv2.imshow('image', img)
    cv2.waitKey(1)