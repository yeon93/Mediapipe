import cv2
import mediapipe as mp
import time
import math
 

class HandDetector():
    def __init__(self, static_img_mode=False, max_num_hands=2, model_complexity=1, 
                 min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_img_mode = static_img_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
         
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_img_mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_conf, self.min_tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
 
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #BGR img => RGV img
        self.results = self.hands.process(imgRGB)        
        # print(results.multi_hand_landmarks)
    
        if self.results.multi_hand_landmarks:    #draw hand landmarks on image
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                               mp.solutions.drawing_styles.get_default_hand_connections_style())
                    
        return img
    
    def find_position(self, img, hand_num=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lm_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx,cy), 8, (255, 0, 255), cv2.FILLED)
    
        return lm_list
    
    def findPosition(self, img, hand_num=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(xList,default="Empty"), max(xList,default="Empty")
        ymin, ymax = min(yList,default="Empty"), max(yList,default="Empty")
        bbox = xmin, ymin, xmax, ymax
        
        return self.lm_list, bbox
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0]-1][1]: 
            fingers.append(1)
        else:
            fingers.append(0)
    
        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id]-2][2]: 
                fingers.append(1)
            else:
                fingers.append(0)
    
            # totalFingers = fingers.count(1)
    
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True, radius=15, thickness=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2
    
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), thickness)
            cv2.circle(img, (x1, y1), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), radius, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2-x1, y2-y1)
    
        return length, img, [x1, y1, x2, y2, cx, cy]
    
    
    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img, draw=False)
        lm_list = detector.find_position(img, hand_num=0, draw=True)
        if len(lm_list) != 0:
            print(lm_list[4])    #position of thumb
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,255), 3) 
        
        cv2.imshow('Image', img)    #cv2.flip(img,1) 1:좌우반전, 0:상하반전
        cv2.waitKey(1) 
    
if __name__ == '__main__':
    main()