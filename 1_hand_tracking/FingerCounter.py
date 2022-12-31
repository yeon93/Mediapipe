import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.HandDetector(max_num_hands=2, min_detection_conf=0.75)
tipIds = [(4,2), (8,6), (12,10), (16,14), (20,18)]
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0][0]][1] > lmList[tipIds[0][1]][1]:            #엄지 손끝(4)의 x좌표가 엄지 뿌리(2)보다 작으면(왼쪽으로 가면)
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id][0]][2] < lmList[tipIds[id][1]][2]:      #손끝의 y좌표가 두번째마디보다 작으면(낮아지면)
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        
        #finger count
        h, w, c = img.shape
        cv2.rectangle(img, (20, 20), (150, 180), (0, 255, 0), 3)
        cv2.putText(img, str(totalFingers), (20, 175), cv2.FONT_HERSHEY_PLAIN,
                    13, (255, 0, 0), 13)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)