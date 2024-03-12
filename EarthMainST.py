import math as m
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt


import time
import os
import mimetypes
from tkinter import *  
from tkinter import messagebox  
from tkinter import filedialog
import pyautogui as pgi

from appcopy import *
from MainTKTest import *

import streamlit as st
import altair

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

mimetypes.init()
# root=Tk()
# variable1=StringVar()    
# variable2=StringVar()    

# root.geometry("800x800")



# Assuming you have a function you want to call
def my_function(var1, var2):
    st.write(f"Variable 1: {var1}")
    st.write(f"Variable 2: {var2}")

# l1 =Label(root, text = "Biomechanical Posture", font= ('Helvetica 25 bold')).place(relx=.5, rely=0,anchor= N)
# l2 =Label(root, textvariable = variable1, font= ('Helvetica 10 bold')).place(relx=.5, rely=.6,anchor= N)
# l3 =Label(root, textvariable = variable2, font= ('Helvetica 10 bold')).place(relx=.5, rely=.7,anchor= N)

# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
pose_video = mp_pose.Holistic(static_image_mode=False, min_detection_confidence=0., model_complexity=1)

def sendWarning(x):
    pass

# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0
# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX
# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
yellow = (0, 255, 255)
pink = (255, 0, 255)
white = (255,255,255)
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


def video_pose_estimation(name):

    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier1 = KeyPointClassifier1()
    point_history_classifier1 = PointHistoryClassifier1()

    keypoint_classifier2 = KeyPointClassifier2()
    point_history_classifier2 = PointHistoryClassifier2()

    # Read labels ###########################################################
    # Wrist twisting
    with open('model1/keypoint_classifier/keypoint_classifier_label_wrist_twist.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels1 = csv.reader(f)
        keypoint_classifier_labels1 = [
            row[0] for row in keypoint_classifier_labels1
        ]
    with open('model1/point_history_classifier/point_history_classifier_label_wrist_twist.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels1 = csv.reader(f)
        point_history_classifier_labels1 = [
            row[0] for row in point_history_classifier_labels1
        ]

    # wrist bending
    with open('model2/keypoint_classifier/keypoint_classifier_label_wrist_bend.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels2 = csv.reader(f)
        keypoint_classifier_labels2 = [
            row[0] for row in keypoint_classifier_labels2
        ]
    with open('model2/point_history_classifier/point_history_classifier_label_wrist_bend.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels2 = csv.reader(f)
        point_history_classifier_labels2 = [
            row[0] for row in point_history_classifier_labels2
        ]
    

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    hand_sign_id1 = 0
    hand_sign_id2 = 0

    camera_video = cv2.VideoCapture(name)
    # Iterate until the webcam is accessed successfully.
    while True:
        fps = cvFpsCalc.get()

        # Camera capture #####################################################
        #ret, image = cap.read()

        # Read a frame.
        ok, frame = cap.read()
        
        # Check if frame is not read properly.
        if not ok:
            
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (960 / frame_height)), 960))

        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        debug_image.flags.writeable = False
        results = hands.process(debug_image)
        debug_image.flags.writeable = True
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

        # debug_image1 = debug_image
        # debug_image2 = debug_image

        # print(frame.shape)
        t1 = time.time()
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)
        # Get fps.
        # fps = camera_video.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        frame_height, frame_width = frame.shape[:2]
        # Convert the BGR frame to RGB.
        
        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame.
        keypoints = pose.process(frame)
        # Convert the frame back to BGR.

        

        # frame.flags.writeable = False
        # results = hands.process(frame)
        # frame.flags.writeable = True

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        global hand_sign_id1L, hand_sign_id1R
        hand_sign_id1L = 3
        hand_sign_id1R = 0

        # Check if the landmarks are detected.
        if landmarks:
            # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=False)   
            
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)

                    # Hand sign classification
                    hand_sign_id1 = keypoint_classifier1(pre_processed_landmark_list)
                    if 0 <= hand_sign_id1 <= 2:
                        hand_sign_id1R = hand_sign_id1
                        # print("hand sign id1 = " + str(hand_sign_id1R))
                    elif 3 <= hand_sign_id1 <= 5:
                        hand_sign_id1L = hand_sign_id1
                        # print("hand sign id1 = " + str(hand_sign_id1L))
                        
                    hand_sign_id2 = keypoint_classifier2(pre_processed_landmark_list)
                    # print("hand sign id2 = " + str(hand_sign_id2)) 

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier1(
                            pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)

                    # debug_image1 = draw_bounding_rect(use_brect, debug_image1, brect)
                    # debug_image1 = draw_landmarks(debug_image1, landmark_list)

                    # debug_image2 = draw_bounding_rect(use_brect, debug_image2, brect)
                    # debug_image2 = draw_landmarks(debug_image2, landmark_list)
                    
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        # wrist twist
                        keypoint_classifier_labels1[hand_sign_id1],
                        point_history_classifier_labels1[most_common_fg_id[0][0]],
                    )

                    # debug_image1 = draw_info_text(
                    #     debug_image1,
                    #     brect,
                    #     handedness,
                    #     # wrist twist
                    #     keypoint_classifier_labels1[hand_sign_id1],
                    #     point_history_classifier_labels1[most_common_fg_id[0][0]],
                    # )

                    # debug_image2 = draw_info_text(
                    #     debug_image2,
                    #     brect,
                    #     handedness,
                    #     # wrist twist
                    #     keypoint_classifier_labels2[hand_sign_id2],
                    #     point_history_classifier_labels2[most_common_fg_id[0][0]],
                    # )
                    
                    # print("hand sign id1 = " + str(hand_sign_id1))
                    # print("hand sign id2 = " + str(hand_sign_id2))
            else:
                point_history.append([0, 0])
                # print("hand sign id1 = -1")
                # print("hand sign id2 = -1")
                hand_sign_id1 = -1
                hand_sign_id1R = -1
                hand_sign_id1L = -1
                hand_sign_id2 = -1
            
        debug_image = draw_info(debug_image, fps)
        frame = draw_info(frame, fps)
        # debug_image1 = draw_info(debug_image1, fps)
        # debug_image2 = draw_info(debug_image2, fps)

        cv2.imshow('Pose Classification', frame)
        # cv2.imshow('Pose Classification', debug_image1)
        # cv2.imshow('Pose Classification', debug_image2)

        # L_wrist_twist_score = find_rula_opp()
        # variable1.set("L_wrist_twist_score = " + str(L_wrist_twist_score))
        
        # variable1.set("hand_sign_id: " + str(hand_sign_id))
        # print("hand_sign_id = " + str(hand_sign_id))

        score_wrist_twist(hand_sign_id1L, hand_sign_id1R)

        L_wrist_bend_score, R_wrist_bend_score = score_wrist_bend(hand_sign_id2)
        wrist_range_score_cal(L_wrist_bend_score)
        


        LC, RC = find_rula_opp('TableA.csv','TableB.csv','TableC.csv')
        
        st.write("LC = " + str(LC) + " and RC = " + str(RC))

        # variable1.set("Left RULA grand score: " + str(LC))
        # variable2.set("Right RULA grand score: " + str(RC))

        # root.update()

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF
        # Check if 'ESC' is pressed.
        if k == 27 and landmarks:
        # Break the loop.
            break
        
    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()

def image_pose_estimation(file_path):
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier1 = KeyPointClassifier1()
    point_history_classifier1 = PointHistoryClassifier1()

    keypoint_classifier2 = KeyPointClassifier2()
    point_history_classifier2 = PointHistoryClassifier2()

    # Read labels ###########################################################
    # Wrist twisting
    with open('model1/keypoint_classifier/keypoint_classifier_label_wrist_twist.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels1 = csv.reader(f)
        keypoint_classifier_labels1 = [
            row[0] for row in keypoint_classifier_labels1
        ]
    with open('model1/point_history_classifier/point_history_classifier_label_wrist_twist.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels1 = csv.reader(f)
        point_history_classifier_labels1 = [
            row[0] for row in point_history_classifier_labels1
        ]

    # wrist bending
    with open('model2/keypoint_classifier/keypoint_classifier_label_wrist_bend.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels2 = csv.reader(f)
        keypoint_classifier_labels2 = [
            row[0] for row in keypoint_classifier_labels2
        ]
    with open('model2/point_history_classifier/point_history_classifier_label_wrist_bend.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels2 = csv.reader(f)
        point_history_classifier_labels2 = [
            row[0] for row in point_history_classifier_labels2
        ]
    

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    hand_sign_id1 = 0
    hand_sign_id2 = 0

    camera_video = cv2.VideoCapture(0)
    # Iterate until the webcam is accessed successfully.
    while True:
        fps = cvFpsCalc.get()

        # Camera capture #####################################################
        #ret, image = cap.read()

        # Read a frame.
        ok, frame = cap.read()
        
        # Check if frame is not read properly.
        if not ok:
            
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.imread(file_path)
        debug_image = copy.deepcopy(frame)

        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (960 / frame_height)), 960))

        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        debug_image.flags.writeable = False
        results = hands.process(debug_image)
        debug_image.flags.writeable = True
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

        # debug_image1 = debug_image
        # debug_image2 = debug_image

        # print(frame.shape)
        t1 = time.time()
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)
        # Get fps.
        # fps = camera_video.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        frame_height, frame_width = frame.shape[:2]
        # Convert the BGR frame to RGB.
        
        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame.
        keypoints = pose.process(frame)
        # Convert the frame back to BGR.

        

        # frame.flags.writeable = False
        # results = hands.process(frame)
        # frame.flags.writeable = True

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        global hand_sign_id1L, hand_sign_id1R
        hand_sign_id1L = 3
        hand_sign_id1R = 0

        # Check if the landmarks are detected.
        if landmarks:
            # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=False)   
            
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)

                    # Hand sign classification
                    hand_sign_id1 = keypoint_classifier1(pre_processed_landmark_list)
                    if 0 <= hand_sign_id1 <= 2:
                        hand_sign_id1R = hand_sign_id1
                        # print("hand sign id1 = " + str(hand_sign_id1R))
                    elif 3 <= hand_sign_id1 <= 5:
                        hand_sign_id1L = hand_sign_id1
                        # print("hand sign id1 = " + str(hand_sign_id1L))
                        
                    hand_sign_id2 = keypoint_classifier2(pre_processed_landmark_list)
                    # print("hand sign id2 = " + str(hand_sign_id2)) 

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier1(
                            pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)

                    # debug_image1 = draw_bounding_rect(use_brect, debug_image1, brect)
                    # debug_image1 = draw_landmarks(debug_image1, landmark_list)

                    # debug_image2 = draw_bounding_rect(use_brect, debug_image2, brect)
                    # debug_image2 = draw_landmarks(debug_image2, landmark_list)
                    
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        # wrist twist
                        keypoint_classifier_labels1[hand_sign_id1],
                        point_history_classifier_labels1[most_common_fg_id[0][0]],
                    )

                    # debug_image1 = draw_info_text(
                    #     debug_image1,
                    #     brect,
                    #     handedness,
                    #     # wrist twist
                    #     keypoint_classifier_labels1[hand_sign_id1],
                    #     point_history_classifier_labels1[most_common_fg_id[0][0]],
                    # )

                    # debug_image2 = draw_info_text(
                    #     debug_image2,
                    #     brect,
                    #     handedness,
                    #     # wrist twist
                    #     keypoint_classifier_labels2[hand_sign_id2],
                    #     point_history_classifier_labels2[most_common_fg_id[0][0]],
                    # )
                    
                    # print("hand sign id1 = " + str(hand_sign_id1))
                    # print("hand sign id2 = " + str(hand_sign_id2))
            else:
                point_history.append([0, 0])
                # print("hand sign id1 = -1")
                # print("hand sign id2 = -1")
                hand_sign_id1 = -1
                hand_sign_id1R = -1
                hand_sign_id1L = -1
                hand_sign_id2 = -1
            
        debug_image = draw_info(debug_image, fps)
        frame = draw_info(frame, fps)
        # debug_image1 = draw_info(debug_image1, fps)
        # debug_image2 = draw_info(debug_image2, fps)

        cv2.imshow('Pose Classification', frame)
        # cv2.imshow('Pose Classification', debug_image1)
        # cv2.imshow('Pose Classification', debug_image2)

        # L_wrist_twist_score = find_rula_opp()
        # variable1.set("L_wrist_twist_score = " + str(L_wrist_twist_score))
        
        # variable1.set("hand_sign_id: " + str(hand_sign_id))
        # print("hand_sign_id = " + str(hand_sign_id))

        score_wrist_twist(hand_sign_id1L, hand_sign_id1R)

        L_wrist_bend_score, R_wrist_bend_score = score_wrist_bend(hand_sign_id2)
        wrist_range_score_cal(L_wrist_bend_score)
        

        global LC, RC
        LC, RC = find_rula_opp('TableA.csv','TableB.csv','TableC.csv')

        st.write("LC = " + str(LC) + " and RC = " + str(RC))
        
        # variable1.set("Left RULA grand score: " + str(LC))
        # variable2.set("Right RULA grand score: " + str(RC))

        # root.update()

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF
        # Check if 'ESC' is pressed.
        if k == 27 and landmarks:
        # Break the loop.
            break
        
    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()

def webcam():
    video_pose_estimation(0)

def browsefunc():
   
   filename =filedialog.askopenfilename()
   mimestart = mimetypes.guess_type(str(filename))[0]

   if mimestart != None:
      mimestart = mimestart.split('/')[0]

   if mimestart == 'video':
      video_pose_estimation(str(filename))
   elif mimestart == 'image':
      image_pose_estimation(str(filename))
   else:
      pass
   
def start(logic):
    if logic == 1:
        if st.button("Choose Live Posture Analysis using webcam"):
            webcam()
        elif st.button("Browse for a video or an audio"):
            browsefunc()
    else:
        pass 

conditionMuscle()
conditionWeight()
global a
a = checkUserInput()

if st.button("Reset"):
    if a == 1:
        a = 0
        start(a)
        a = 1

st.write("")
st.write("")
st.write("")
st.write("<span style='font-weight: bold; color: rgb(255, 180, 10); font-size: 16px; '>Please check if the condtions are provided correctly before starting the application!</span>", unsafe_allow_html=True)

start(a)
