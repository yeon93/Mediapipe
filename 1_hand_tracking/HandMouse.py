import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

###############################
wCam, hCam = 640, 480
frameR = 100      #Frame Reduction
smoothening = 7   #커서 움직임
###############################

pTime = 0
prev_loc_X, prev_loc_Y = 0, 0
curr_loc_X, curr_loc_Y = 0, 0

cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(max_num_hands=1)
wScreen, hScreen = autopy.screen.size()
# print(wScreen, hScreen)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list, bbox = detector.findPosition(img)
    
    # 2. Get the tip of the index and middle fingers
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        # print(x1, y1, x2, y2)

    # 3. 펼쳐진 손가락 확인
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR),
                      (255, 0, 255), 2)
    
    # 4. 이동 모드 : 검지만 폈을 때
        if fingers[1] == 1 and fingers[2] == 0:
        
    # 5. 좌표 변환    
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScreen))
    
    # 6. Smoothen Values
            curr_loc_X = prev_loc_X + (x3-prev_loc_X) / smoothening
            curr_loc_Y = prev_loc_Y + (y3-prev_loc_Y) / smoothening

    # 7. 마우스 이동
            autopy.mouse.move(wScreen-curr_loc_X, curr_loc_Y)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            prev_loc_X, prev_loc_Y = curr_loc_X, curr_loc_Y

    # 8. 클릭 모드 : 검지 & 중지를 폈을 때 거리가 40 미만이면 클릭 실행
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)    #손가락 사이 거리 계산
            #print(length)
            if length < 40:    
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame Rate
    #cTime = time.time()
    #fps = 1 / (cTime-pTime)
    #pTime = cTime
    #cv2.putText(img, f'FPS:{str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    #            (255, 0, 0), 3)
    
    # 12. Display
    cv2.imshow('Image', cv2.flip(img, 1))
    cv2.waitKey(1)