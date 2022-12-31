import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushThickness = 25
eraserThickness = 100
########################

#https://www.canva.com/ 에서 헤더 디자인
folderPath = 'C:\\Users\\yeonok\\Desktop\\projects\\Computer vision\\1_hand_tracking\\Header'
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(min_detection_conf=0.75, max_num_hands=1)
xp, yp = 0, 0    #previous position
imgCanvas = np.zeros((720, 1280, 3), np.uint8)    #color img

while True:
    # 1. Import image
    success, img = cap.read()
    #img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.find_hands(img)
    lm_list, bbox = detector.findPosition(img, draw=False)

    if len(lm_list) != 0:
        # print(lm_list)
        x1, y1 = lm_list[8][1:]    #검지손가락 끝
        x2, y2 = lm_list[12][1:]   #중지손가락 끝

    # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

    # 4. If Selection Mode – Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('Selection Mode')
            
            # Checking for the click
            if y1 < 125:     #손끝이 화면의 상단에 위치하면
                if 250 < x1 < 450:                #빨간 붓 선택
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 550 < x1 < 750:              #파란 붓 선택
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:              #노란 붓 선택
                    header = overlayList[2]
                    drawColor = (0, 255, 255)
                elif 1050 < x1 < 1200:            #지우개 선택
                    header = overlayList[3]
                    drawColor = (0, 0, 0)         #black
                    
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode – Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('Drawing Mode')
            
            if xp == 0 and yp == 0:    #처음 손을 인식했을 때
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):    #지우개 모드일 때
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

        # Clear Canvas when all fingers are up
        if all (x >= 1 for x in fingers):  
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:150, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)    #img, imgCanvas를 동시에 보여줌
    cv2.imshow('Image', img)
    #cv2.imshow('Canvas', imgCanvas)
    #cv2.imshow('Inv', imgInv)
    cv2.waitKey(1)