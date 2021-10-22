import cv2
import mediapipe as mp
import time

mp_Draw = mp.solutions.drawing_utils
mp_Pose = mp.solutions.pose
pose    = mp_Pose.Pose()

cap = cv2.VideoCapture('pushup.mp4')
pTime = 0

while True:
   
    success, img = cap.read()
    results      = pose.process(img)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_Draw.draw_landmarks(img, results.pose_landmarks, mp_Pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w ,c = img.shape
            # print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps   = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow('image', img)
    
    key = cv2.waitKey(10)
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
