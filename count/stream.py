import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp


class poseDetector():
 
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
 
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
 
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
 
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
 
    def findAngle(self, img, p1, p2, p3, draw=True):
 
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
 
        # Calculate the Angle
        # angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
        #                      math.atan2(y1 - y2, x1 - x2))
        # if angle < 0:
        #     angle += 360
        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle   = np.abs(radians*180/np.pi) 

        if angle >180:
            angle = 360 - angle
        # print(angle)
 
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            # cv2.putText(img, str(int(angle)), (x2 - 80, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), tuple(np.add(self.lmList[p2][1:],[20,20]).astype(int) ),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        
        return angle

detector = poseDetector()
count = 0
dir   = 0
pTime = 0
state = None


st.title('Fitness calaorie counter app')
html_temp = '''
<body style = 'background-color: red,'>
<div style = 'background-color:teal , padding:10px'>
<h2 style = 'color:white; text-align,'>Fitness Tracker webapp</h2>
</div>
</body>
'''

st.markdown(html_temp, unsafe_allow_html=True)


start = st.button('Start Training')
stop  = st.button('Stop Training')
frame_window = st.image([])
cap = cv2.VideoCapture(0)
while start:  
    ret, img = cap.read()
    img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # x   = frame_window.image(img1)
    img = detector.findPose(img,False)
    height = img.shape[0]
    width  = img.shape[1]

    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        angle1 = detector.findAngle(img, 11, 13, 15) # Left  hand
        angle2 = detector.findAngle(img, 12, 14, 16) #Right hand
        # angle1 = detector.findAngle(img,11,23,25, True) #situps

        # angle1 = detector.findAngle(img,0,11,23)

        per = np.interp(angle1,(70,160), (0,100)) #checking the angles
        # bar = np.interp(angle1, (190,280), (650,100))
        # print(bar)
      
        # if(lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]) and dir ==0:
        #     dir ==1      
        #     count += 0.5

        # if(lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and dir ==1:  
        #     count += 1
        #     dir == 0 
        # print(count)


        # check the angle reaches 100 to convert a curve
        if per >= 70:
            if dir == 0:
                count +=0.5
                dir = 1
        if per <= 40:
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)
        cv2.rectangle(img, (width-100 , 0), (width , 100), (0,0,0), -1) #count box
        cv2.putText(img, str(int(count)), (width-100,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5) #count text
        cv2.putText(img, 'Counts', (width-90,35), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2) # count written text

        cv2.rectangle(img, (width-130 , 0), (width-110 , 100), (0,0,0), 3) # barometer
        cv2.rectangle(img, (width-128 , int(per) ), (width-112 , 0), (255,0,0), -1) # barometer interior
        cv2.putText(img, f'{int(per)} %', (width-130,120), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2) # count written text

    cTime = time.time()
    fps   = 1/(cTime - pTime)
    pTime = cTime 
    cv2.putText(img, str(int(fps)), (30,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    frame_window.image(img)
    key = cv2.waitKey(10)
    if key == stop:
        break

