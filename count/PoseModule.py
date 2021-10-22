
import cv2
import mediapipe as mp
import time
import math
import numpy as np
 
 
class poseDetector():
 
    def __init__(self, mode=False, model_complex = 1,smooth_landmarks=True, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
 
        self.mode = mode
        self.model_complex = model_complex
        self.smooth_landmarks = smooth_landmarks
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complex,self.smooth_landmarks, self.upBody, self.smooth)
                                     #self.detectionCon, self.trackCon)
 
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
 
def main():
    cap = cv2.VideoCapture('videos/pushup.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList)
            
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (lmList[13][1], lmList[13][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
 
        cv2.imshow("Image", img)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break

def ManualFindPose(img_input):
    detector = poseDetector()
    img = detector.findPose(img_input, draw=False)
    lmList = detector.findPosition(img, draw=False)

    mask  = np.zeros_like(img)

    # Define the Nodes with Red Dots
    

    #Draw the Lines between Nodes
    # Nose to Shoulders
    cv2.line(mask, (lmList[0][1],lmList[0][2]), (lmList[11][1],lmList[11][2]), (0, 255, 0), 2)
    cv2.line(mask, (lmList[0][1],lmList[0][2]), (lmList[12][1],lmList[12][2]), (0, 255, 0), 2)
    # Wrist to Elbow
    cv2.line(mask, (lmList[16][1],lmList[16][2]), (lmList[14][1],lmList[14][2]), (0, 255, 0), 2)
    cv2.line(mask, (lmList[15][1],lmList[15][2]), (lmList[13][1],lmList[13][2]), (0, 255, 0), 2)
    # Elbow to Shoulder
    cv2.line(mask, (lmList[14][1],lmList[14][2]), (lmList[12][1],lmList[12][2]), (0, 255, 0), 2)
    cv2.line(mask, (lmList[13][1],lmList[13][2]), (lmList[11][1],lmList[11][2]), (0, 255, 0), 2)
    # Shoulder to Shoulder
    cv2.line(mask, (lmList[11][1],lmList[11][2]), (lmList[12][1],lmList[12][2]), (0, 255, 0), 2)
    # Shoulder to Hip
    cv2.line(mask, (lmList[11][1],lmList[11][2]), (lmList[23][1],lmList[23][2]), (0, 255, 0), 2)
    cv2.line(mask, (lmList[24][1],lmList[24][2]), (lmList[12][1],lmList[12][2]), (0, 255, 0), 2)
    # Hip to Hip
    cv2.line(mask, (lmList[24][1],lmList[24][2]), (lmList[23][1],lmList[23][2]), (0, 255, 0), 2)
    # Hip to Knee
    cv2.line(mask, (lmList[24][1],lmList[24][2]), (lmList[26][1],lmList[26][2]), (0, 255, 0), 2)
    cv2.line(mask, (lmList[25][1],lmList[25][2]), (lmList[23][1],lmList[23][2]), (0, 255, 0), 2)
    # Knee to Ankle
    cv2.line(mask, (lmList[28][1],lmList[28][2]), (lmList[26][1],lmList[26][2]), (0, 255, 0), 2)
    cv2.line(mask, (lmList[25][1],lmList[25][2]), (lmList[27][1],lmList[27][2]), (0, 255, 0), 2)

    # Node 0 : Nose
    cv2.circle(mask, (lmList[0][1], lmList[0][2]), 5, (203,192,255), cv2.FILLED)

    # Node 15 & 16 : Wrist
    cv2.circle(mask, (lmList[15][1], lmList[15][2]), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(mask, (lmList[16][1], lmList[16][2]), 5, (0, 0, 255), cv2.FILLED)
            
    # Node 13 & 14 : Elbow
    cv2.circle(mask, (lmList[14][1], lmList[14][2]), 5, (0, 140, 255), cv2.FILLED)
    cv2.circle(mask, (lmList[13][1], lmList[13][2]), 5, (0, 140, 255), cv2.FILLED)

    # Node 11 & 12 : Shoulder
    cv2.circle(mask, (lmList[11][1], lmList[11][2]), 5, (0, 255, 255), cv2.FILLED)
    cv2.circle(mask, (lmList[12][1], lmList[12][2]), 5, (0, 255, 255), cv2.FILLED)

    # Node 23 & 24 : Hip
    cv2.circle(mask, (lmList[23][1], lmList[23][2]), 5, (0, 255, 255), cv2.FILLED)
    cv2.circle(mask, (lmList[24][1], lmList[24][2]), 5, (0, 255, 255), cv2.FILLED)

    # Node 25 & 26 : Knee
    cv2.circle(mask, (lmList[25][1], lmList[25][2]), 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(mask, (lmList[26][1], lmList[26][2]), 5, (255, 0, 0), cv2.FILLED)

    # Node 27 & 28 : Ankle
    cv2.circle(mask, (lmList[27][1], lmList[27][2]), 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(mask, (lmList[28][1], lmList[28][2]), 5, (255, 0, 0), cv2.FILLED)

    return mask
        
        
if __name__ == "__main__":
    main()
