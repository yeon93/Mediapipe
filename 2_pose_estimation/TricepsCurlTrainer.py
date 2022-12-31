import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY)
detector = pm.poseDetector()

count = 0    
direction = 0      #0:팔을 굽힐 때, 1:펼 때 
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    # img = cv2.imread("")
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        angle = detector.findAngle(img, 12, 14, 16)     #angle of right arm(shoulder, elbow, wrist)
        #angle = detector.findAngle(img, 11, 13, 15)    #left arm
        
        #np.interp(x, xp, fp, ..) : One-dimensional linear interpolation for monotonically increasing sample points
        per = np.interp(angle, (210, 310), (0, 100))      #angle이 210-310일 때 0-100으로 보간
        bar = np.interp(angle, (220, 310), (400, 100))
        # print(angle, per)
        
        # Check for the triceps curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if direction == 0:    #굽힐 때
                count += 0.5
                direction = 1
        if per == 0:
            color = (0, 255, 0)
            if direction == 1:    #펼 때
                count += 0.5
                direction = 0
        print(count)
        
        # Draw Bar
        cv2.rectangle(img, (580, 100), (600, 400), color, 3)
        cv2.rectangle(img, (580, int(bar)), (600, 400), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (555, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                    color, 2)
        
        # Draw Curl Count
        cv2.rectangle(img, (0, 320), (200, 480), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (10, 450), cv2.FONT_HERSHEY_PLAIN, 8,
                    (255, 255, 255), 10)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)