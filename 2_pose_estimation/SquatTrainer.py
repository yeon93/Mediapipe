import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY)
detector = pm.poseDetector()

count = 0    
direction = 0      #0:굽힐 때, 1:펼 때 
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    # img = cv2.imread("")
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        angle_img = detector.findAngle(img, 24, 26, 28)     #angle of right leg(hip, knee, ankle)
        #angle_img = detector.findAngle(img, 23, 25, 27)    #left leg
        
        #np.interp(x, xp, fp, ..) : One-dimensional linear interpolation for monotonically increasing sample points
        #x : array_like, The x-coordinates at which to evaluate the interpolated values.
        #xp : 1-D sequence of floats, The x-coordinates of the data points, must be increasing if argument period is not specified.
        # Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
        #fp : 1-D sequence of float or complex, The y-coordinates of the data points, same length as xp.
        per = np.interp(angle_img, (210, 310), (0, 100))
        bar = np.interp(angle_img, (220, 310), (650, 100))
        # print(angle_img, per)
        
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
        cv2.putText(img, f'{int(per)} %', (575, 80), cv2.FONT_HERSHEY_PLAIN, 3,
                    color, 3)
        
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