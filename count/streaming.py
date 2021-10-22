import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import torch
import PoseModule as pm
import Pose_classifier as pc
from PIL import Image
st.set_page_config(layout = 'wide')

# ------------------------------------ START OF DEEP LEARNING LOADING PART -------------------------------------------------------------

# Pose Detector & Pose Classifier

detector = pm.poseDetector()
count = 0
dir   = 0
pTime = 0
state = None
workout_start_time = time.time()

#Loading Model
checkpoint=torch.load('Best_Pose_Model_7.pth')  # replace your path with the (.pth) file

model = pc.ConvNet(num_classes=2)
model.load_state_dict(checkpoint)
model.eval()

#Variables for This internal file use
time_skip = 250 # in miliseconds
time_skip_seconds = 0.50 # in seconds
frame_skip = 60
start_time = time.time()
pose_list = []
constant_pose_bool=False
counter = 0

# ------------------------------------ END OF DEEP LEARNING LOADING PART -------------------------------------------------------------

# ------------------------------------ START OF STREAMLIT PART -------------------------------------------------------------

st.sidebar.title('Hey Champ 💪')
st.sidebar.subheader('Please fill the details😎')
# st.sidebar.subheader('Gender')
selected_category = st.sidebar.selectbox('Gender', ('Women', 'Man'))
age = st.sidebar.number_input('Please enter your age',18)
height = st.sidebar.number_input('Please enter your height in cms',160)
weight = st.sidebar.number_input('Please enter your Weight in kgs',60)
st.sidebar.write('---------------------')
def lbs_2_kg(weight):
    weight_in_kg = weight / 2.2
    return int(weight_in_kg)

def feet_2_cms(height):  
    height_in_inch = height * 12 * 2.54
    return (height_in_inch)

convert_height = st.sidebar.number_input('Convert inch to cms', 5.5 )
height_inch = feet_2_cms(convert_height)
st.sidebar.write(f'Your Height in cms is {height_inch: .2f}')   

convert_weight = st.sidebar.number_input('Convert pounds to Kgs', 140 )
weight_kg = lbs_2_kg(convert_weight)
st.sidebar.write(f'Your weight in kgs is {weight_kg: .2f}')  


st.title('Fitness Trainer Web App')
html_temp = '''
<body style = 'background-color: red,'>
<div style = 'background-color:teal , padding:10px'>
<h2 style = 'color:white; text-align,'>Click start when you are ready</h2>
</div>
</body>
'''
st.markdown(html_temp, unsafe_allow_html=True)


start = st.button('Start Training')
stop  = st.button('Stop Training')
frame_window = st.image([])

# ---------------------------------------------------------------- END OF STREAMLIT PART --------------------------------------------------------
# ---------------------------------------------------------------- START OF MOTION CAPTURE PART --------------------------------------------------------


cap = cv2.VideoCapture(0)
while start:
    # while (cap.isOpened()):
    success, img1 = cap.read()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    
    if success == False:
        break
    
    sk_img  = np.zeros_like(img1)

    current_time = time.time()
    time_lapsed = current_time - start_time

    if constant_pose_bool == False:
        if time_lapsed % time_skip_seconds <= 0.01:
        #if cap.get(0)%time_skip <= 50:
        #if cap.get(1)%frame_skip == 0:
            try:
                img = detector.findPose(img1,False)
                height = img.shape[0]
                width  = img.shape[1]
                lmList = detector.findPosition(img, False)                
                sk_img = pm.ManualFindPose(img)
                img2 = cv2.cvtColor(sk_img, cv2.COLOR_BGR2RGB)
                img_new =Image.fromarray(img2)
                img_classified = pc.prediction(img_new,pc.transformer)
                print('Pose Prediction :', img_classified)
                cv2.putText(img1,f'Detecting: {img_classified}', (0,height - 70), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                #print('elapsed time = ', time_lapsed)
                # cv2.imshow('image_sk', sk_img)

                # Append the List to get 10 consequtive pose
                #print(constant_pose_bool)
                print('List Counter #', counter)
                counter += 1
                pose_list.append(img_classified)
                if len(pose_list)==5:
                    #True Maker
                    constant_pose_bool = all(element == pose_list[0] for element in pose_list)   
                    if constant_pose_bool == True:
                        print(constant_pose_bool)
                        chosen_activity = pose_list[0]
                        pass
                    else:
                        counter = 0
                        pose_list = []
                        cv2.putText(img1, 'Pose not stabilised, Please wait', (0,height - 90), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            except:
                #print('no img detected')
                pass
    else:
        img = detector.findPose(img1,False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) != 0:
            if chosen_activity == 'pushup':
                cv2.putText(img1, 'Push Up', (0,height -35), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                angle1 = detector.findAngle(img, 11, 13, 15) # Left  hand
                angle2 = detector.findAngle(img, 12, 14, 16) # Right hand
            
            else: #If it is sit up
                cv2.putText(img1, 'Sit Up', (0,height - 35), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                angle1 = detector.findAngle(img1,11,23,25) #situps
                angle2 = detector.findAngle(img1,12,24,26) #situps
        
            per = np.interp(angle1,(70,160), (0,100)) #checking the angles
            
            if per >= 90:
                if dir == 0:
                    count +=0.5
                    dir = 1
            if per <= 25:
                if dir == 1:
                    count += 0.5
                    dir = 0
            print(count)
            # bmr_man = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
            # bmr_women = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
            # met = bmr_man * count * time_lapsed/3660
            cal_burnt_per_hour = (3.8 * 3.5 * count ) / 70 
            
            cv2.rectangle(img1, (width-100 , 0), (width , 100), (0,0,0), -1) #count box
            cv2.putText(img1, str(int(count)), (width-100,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5) #count text
            cv2.putText(img1, 'Counts', (width-90,35), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2) # count written text

            cv2.rectangle(img1, (width-130 , 0), (width-110 , 100), (0,0,0), 3) # barometer
            cv2.rectangle(img1, (width-128 , int(per) ), (width-112 , 0), (255,0,0), -1) # barometer interior
            cv2.putText(img1, f'{int(per)} %', (width-130,120), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2) # count written text
            cv2.rectangle(img1, (width-250 , 0), (width-140 , 100), (0,0,0), -1) # barometer
            cv2.putText(img1, str(int(cal_burnt_per_hour)), (width-240, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            cv2.putText(img1, 'Calories', (width-240, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        
        # CHECK INDENTATION LEVEL

            cTime = time.time()
            fps   = 1/(cTime - pTime)
            total_work_out_time = cTime - pTime
            pTime = cTime 

            cv2.putText(img1, str(int(fps)), (30,80), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.putText(img1, (f'Time {time_lapsed: .2f}'), (30,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        
    #Show the Video on Streamlit
    frame_window.image(img1)

    key = cv2.waitKey(10)
    if key == stop:
        break


            

                

    
    
    
    

    #---------------------------------------------------------------------- THE GREAT BARRIER OF MERGING -----------------------------------------------------------
    
