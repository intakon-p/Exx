import cv2 as cv
import mediapipe as mp
import matplotlib
# matplotlib.use('module://ipykernel.pylab.backend_inline')
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from angle_cal import *

import math as m
import cv2
import numpy as np
import time
import mediapipe as mp

import pandas as pd

import altair

import os
import mimetypes

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

from utils import CvFpsCalc

from openpyxl import load_workbook

import tkinter



# Initialize variables
Nose_Pos = None
Left_eye_inner_Pos = None
Left_eye_Pos = None
Left_eye_outer_Pos = None
Right_eye_inner_Pos = None
Right_eye_Pos = None
Right_eye_outer_Pos = None
Left_ear_Pos = None
Right_ear_Pos = None
Mouth_left_Pos = None
Mouth_right_Pos = None
Left_shoulder_Pos = None
Right_shoulder_Pos = None
Left_elbow_Pos = None
Right_elbow_Pos = None
Left_wrist_Pos = None
Right_wrist_Pos = None
Left_pinky_Pos = None
Right_pinky_Pos = None
Left_index_Pos = None
Right_index_Pos = None
Left_thumb_Pos = None
Right_thumb_Pos = None
Left_hip_Pos = None
Right_hip_Pos = None
Left_knee_Pos = None
Right_knee_Pos = None
Left_ankle_Pos = None
Right_ankle_Pos = None
Left_heel_Pos = None
Right_heel_Pos = None
Left_foot_index_Pos = None
Right_foot_index_Pos = None
Neck_Pos = None
Head_Pos = None
Hip_Pos = None

def video_pose_estimation2(name):

    # Initialize MediaPipe pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    Pose = mp_pose.PoseLandmark

    # Define colors for each keypoint
    keypoint_colors = {
        Pose.NOSE: '#dd0053',#0
        Pose.LEFT_EYE_INNER: '#db00c7',#1
        Pose.LEFT_EYE: '#980364',#2
        Pose.LEFT_EYE_OUTER: '#920534',#3
        Pose.RIGHT_EYE_INNER: '#640297',#4
        Pose.RIGHT_EYE: '#66009f',#5
        Pose.RIGHT_EYE_OUTER: '#9a006e',#6
        Pose.LEFT_EAR: '#9c006a',#7
        Pose.RIGHT_EAR: '#950360',#8
        Pose.MOUTH_LEFT: '#9a006e',#9
        Pose.MOUTH_RIGHT: '#9a006e',#10
        Pose.LEFT_SHOULDER: '#329c03',#11
        Pose.RIGHT_SHOULDER: '#9c630a',#12
        Pose.LEFT_ELBOW: '#029b01',#13
        Pose.RIGHT_ELBOW: '#999700',#14
        Pose.LEFT_WRIST: '#019934',#15
        Pose.RIGHT_WRIST: '#679800',#16
        Pose.LEFT_PINKY: 'gray',#17
        Pose.RIGHT_PINKY: 'gray',#18
        Pose.LEFT_INDEX: 'gray',#19
        Pose.RIGHT_INDEX: 'gray',#20
        Pose.LEFT_THUMB: 'gray',#21
        Pose.RIGHT_THUMB: 'gray',#22
        Pose.LEFT_HIP: '#003297',#23
        Pose.RIGHT_HIP: '#049a66',#24
        Pose.LEFT_KNEE: '#0d0099',#25
        Pose.RIGHT_KNEE: '#039692',#26
        Pose.LEFT_ANKLE: '#330098',#27
        Pose.RIGHT_ANKLE: '#016396',#28
        Pose.LEFT_HEEL: 'gray',#29
        Pose.RIGHT_HEEL: 'gray',#30
        Pose.LEFT_FOOT_INDEX: 'gray',#31
        Pose.RIGHT_FOOT_INDEX: 'gray',#32
        
        # lime olive magenta navy indigo gold teal maroon
    }

    # Define colors for each line (connection) between keypoints
    connection_colors = {
        (Pose.LEFT_SHOULDER, Pose.RIGHT_SHOULDER): '#ce4802',
        (Pose.LEFT_SHOULDER, Pose.LEFT_ELBOW): '#679702',
        (Pose.LEFT_SHOULDER, Pose.LEFT_HIP): '#04989a',
        
        (Pose.RIGHT_SHOULDER, Pose.RIGHT_ELBOW): '#9d6309',
        (Pose.RIGHT_SHOULDER, Pose.RIGHT_HIP): '#019900',

        (Pose.RIGHT_ELBOW, Pose.RIGHT_WRIST): '#999b00',
        (Pose.LEFT_ELBOW, Pose.LEFT_WRIST): '#2e9b01',

        (Pose.LEFT_HIP, Pose.RIGHT_HIP): '#029988',

        (Pose.RIGHT_HIP, Pose.RIGHT_KNEE): '#009b32',
        (Pose.LEFT_HIP, Pose.LEFT_KNEE): '#00669a',

        (Pose.RIGHT_KNEE, Pose.RIGHT_ANKLE): '#003496',
        (Pose.LEFT_KNEE, Pose.LEFT_ANKLE): '#049866',

        (Pose.LEFT_SHOULDER, Pose.LEFT_ANKLE): '#049866',

        # Define other connections and their colors as needed
    }

    # Define indices of keypoints for drawing lines (connections)
    keypoint_connections = [
        (11, 13),  # Left shoulder to left elbow
        (13, 15),  # Left elbow to left wrist
        (12, 14),  # Right shoulder to right elbow
        (14, 16),  # Right elbow to right wrist
        (11, 23),  # Left shoulder to left hip
        (12, 24),  # Right shoulder to right hip
        (23, 24),  # Left hip to right hip
        (23, 25),  # Left hip to left knee
        (24, 26),  # Right hip to right knee
        (25, 27),  # Left knee to left ankle
        (26, 28)   # Right knee to right ankle
    ]

    # Create a new figure for subplots
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    # Set titles and view angles for each subplot
    
    ax1.set_title('Front')
    ax1.view_init(elev=-90, azim=-90, roll=0)
    ax2.set_title('Top')
    ax2.view_init(elev=0, azim=-90, roll=180)
    ax3.set_title('Side')
    ax3.view_init(elev=0, azim=-180, roll=90)

    # Open the webcam
    cap = cv2.VideoCapture(name)

    # Define global variables for keypoints
    global Nose_Pos #0
    global Left_eye_inner_Pos #1
    global Left_eye_Pos
    global Left_eye_outer_Pos
    global Right_eye_inner_Pos
    global Right_eye_Pos #5
    global Right_eye_outer_Pos
    global Left_ear_Pos
    global Right_ear_Pos
    global Mouth_left_Pos
    global Mouth_right_Pos #10
    global Left_shoulder_Pos
    global Right_shoulder_Pos
    global Left_elbow_Pos
    global Right_elbow_Pos
    global Left_wrist_Pos #15
    global Right_wrist_Pos
    global Left_pinky_Pos
    global Right_pinky_Pos
    global Left_index_Pos
    global Right_index_Pos #20
    global Left_thumb_Pos
    global Right_thumb_Pos
    global Left_hip_Pos
    global Right_hip_Pos
    global Left_knee_Pos #25
    global Right_knee_Pos
    global Left_ankle_Pos
    global Right_ankle_Pos
    global Left_heel_Pos
    global Right_heel_Pos #30
    global Left_foot_index_Pos
    global Right_foot_index_Pos #32
    global Neck_Pos #33
    global Head_Pos #34
    global Hip_Pos #35

    global step1_left_score
    global step1_right_score
    global step2_left_score
    global step2_right_score
    global step3_left_score
    global step3_right_score
    global step4_left_score
    global step4_right_score

    global step9_score
    global step10_score
    global step11_score


    image_placeholder = st.empty()
    text_placeholder1 = st.empty()
    text_placeholder2 = st.empty()
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
   
        # frame = cv2.flip(frame, 1)
        frame = cv2.imread('b1.jpg')

        frame_height, frame_width, _ =  frame.shape

        frame = cv2.resize(frame, (int(frame_width * (960 / frame_height)), 960))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(frame)
        
        output_image = frame.copy()

        height, width, _ = frame.shape
    
        # Initialize a list to store the detected landmarks.
        landmarks = []

        time_ = 0
        mp_pose = mp.solutions.holistic
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing = mp.solutions.drawing_utils

        # If pose detected, plot keypoints and lines
        if results.pose_landmarks:
            for ax in [ax1, ax2, ax3]:
                ax.clear()
                # Extract 3D keypoints
                keypoints_3d = []
                for landmark in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z
                    keypoints_3d.append((cx, cy, cz))

                # Calculate midpoint between left and right shoulders
                left_shoulder = keypoints_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = keypoints_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                midpoint_shoulders = ((keypoints_3d[11][0] + keypoints_3d[12][0]) / 2, (keypoints_3d[11][1] + keypoints_3d[12][1]) / 2, (keypoints_3d[11][2] + keypoints_3d[12][2]) / 2)
                
                # Calculate midpoint between left and right ears
                left_ear = keypoints_3d[mp_pose.PoseLandmark.LEFT_EAR.value] 
                right_ear = keypoints_3d[mp_pose.PoseLandmark.RIGHT_EAR.value] 
                midpoint_ears = ((keypoints_3d[7][0] + keypoints_3d[8][0]) / 2, (keypoints_3d[7][1] + keypoints_3d[8][1]) / 2, (keypoints_3d[7][2] + keypoints_3d[8][2]) / 2)

                # Calculate midpoint between left and right hips
                left_hip = keypoints_3d[mp_pose.PoseLandmark.LEFT_HIP.value] 
                right_hip = keypoints_3d[mp_pose.PoseLandmark.RIGHT_HIP.value]
                midpoint_hips = ((keypoints_3d[23][0] + keypoints_3d[24][0]) / 2, (keypoints_3d[23][1] + keypoints_3d[24][1]) / 2, (keypoints_3d[23][2] + keypoints_3d[24][2]) / 2)

                # Add the midpoint shoulders to keypoints_3d
                keypoints_3d.append(midpoint_shoulders)
                # Add the midpoint shoulders to keypoint_colors
                keypoint_colors[len(keypoints_3d) - 1] = '#ce4802'  # Green color for midpoint shoulders

                # Add the midpoint ears to keypoints_3d
                keypoints_3d.append(midpoint_ears)
                # Add the midpoint ears to keypoint_colors
                keypoint_colors[len(keypoints_3d) - 1] = '#dd0053'  # Green color for midpoint shoulders

                # Add the midpoint hips to keypoints_3d
                keypoints_3d.append(midpoint_hips)
                # Add the midpoint hips to keypoint_colors
                keypoint_colors[len(keypoints_3d) - 1] = '#042a66'  # Green color for midpoint shoulders

                # Define connections between the midpoint and other keypoints
                keypoint_connections_with_midpoint = [
                    # neck to head
                    (len(keypoints_3d) - 3, len(keypoints_3d) - 2, '#0d0096'),
                    # neck to hip
                    (len(keypoints_3d) - 3, len(keypoints_3d) - 1, '#04989a'),
                    
                    # Midpoint shoulders to left shoulder (color: blue)
                    (len(keypoints_3d) - 3, Pose.LEFT_SHOULDER.value, '#d44606'),
                    # Midpoint shoulders to right shoulder (color: red)
                    (len(keypoints_3d) - 3, Pose.RIGHT_SHOULDER.value, '#970102'),
                    # Add more connections as needed
                ]

                # Plot keypoints in 3D space with specified color
                for idx, point_3d in enumerate(keypoints_3d):
                    color = keypoint_colors.get(idx, 'gray')  # Default to gray if color not specified
                    ax.scatter(point_3d[0], point_3d[1], point_3d[2], c=color, marker='o')

                # Draw lines between connected keypoints with specified color
                for connection in keypoint_connections:
                    start_point_idx, end_point_idx = connection
                    start_point = keypoints_3d[start_point_idx]
                    end_point = keypoints_3d[end_point_idx]
                    color = connection_colors.get(connection, 'gray')  # Default to gray if color not specified
                    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=color)

                # Plot lines connecting the midpoint to each shoulder with different colors
                for connection in keypoint_connections_with_midpoint:
                    start_point_idx, end_point_idx, color = connection
                    start_point = keypoints_3d[start_point_idx]
                    end_point = keypoints_3d[end_point_idx]
                    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=color)

                # print("Keypoint positions:")
                for idx, point_3d in enumerate(keypoints_3d):
                    # print(f"Keypoint {idx}: {point_3d}")
                    # Assign values to global variables based on the keypoint index
                    if idx == 0:
                        Nose_Pos = point_3d
                    elif idx == 1:
                        Left_eye_inner_Pos = point_3d
                    elif idx == 2:
                        Left_eye_Pos = point_3d
                    elif idx == 3:
                        Left_eye_outer_Pos = point_3d
                    elif idx == 4:
                        Right_eye_inner_Pos = point_3d
                    elif idx == 5:
                        Right_eye_Pos = point_3d
                    elif idx == 6:
                        Right_eye_outer_Pos = point_3d
                    elif idx == 7:
                        Left_ear_Pos = point_3d
                    elif idx == 8:
                        Right_ear_Pos = point_3d
                    elif idx == 9:
                        Mouth_left_Pos = point_3d
                    elif idx == 10:
                        Mouth_right_Pos = point_3d
                    elif idx == 11:
                        Left_shoulder_Pos = point_3d
                    elif idx == 12:
                        Right_shoulder_Pos = point_3d
                    elif idx == 13:
                        Left_elbow_Pos = point_3d
                    elif idx == 14:
                        Right_elbow_Pos = point_3d
                    elif idx == 15:
                        Left_wrist_Pos = point_3d
                    elif idx == 16:
                        Right_wrist_Pos = point_3d
                    elif idx == 17:
                        Left_pinky_Pos = point_3d
                    elif idx == 18:
                        Right_pinky_Pos = point_3d
                    elif idx == 19:
                        Left_index_Pos = point_3d
                    elif idx == 20:
                        Right_index_Pos = point_3d
                    elif idx == 21:
                        Left_thumb_Pos = point_3d
                    elif idx == 22:
                        Right_thumb_Pos = point_3d
                    elif idx == 23:
                        Left_hip_Pos = point_3d
                    elif idx == 24:
                        Right_hip_Pos = point_3d
                    elif idx == 25:
                        Left_knee_Pos = point_3d
                    elif idx == 26:
                        Right_knee_Pos = point_3d
                    elif idx == 27:
                        Left_ankle_Pos = point_3d
                    elif idx == 28:
                        Right_ankle_Pos = point_3d
                    elif idx == 29:
                        Left_heel_Pos = point_3d
                    elif idx == 30:
                        Right_heel_Pos = point_3d
                    elif idx == 31:
                        Left_foot_index_Pos = point_3d
                    elif idx == 32:
                        Right_foot_index_Pos = point_3d
                    elif idx == 33:
                        Neck_Pos = point_3d
                    elif idx == 34:
                        Head_Pos = point_3d
                    elif idx == 35:
                        Hip_Pos = point_3d

                    # Draw Pose landmarks on the output image.
                    mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # Iterate over the detected landmarks.
                    for landmark in results.pose_landmarks.landmark:
                        # Append the landmark into the list.
                        landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                            (landmark.z * width)))

        # Calibrate
        calibrate_angle = calculate_angle(Left_shoulder_Pos, Right_shoulder_Pos, (Right_shoulder_Pos[0], Right_shoulder_Pos[1],Right_shoulder_Pos[2]-1), 'top')
        if 60 <= calibrate_angle < 120:
            view = "0" # Up
        elif 30 < calibrate_angle < 60:
            view = "1" # Up-Left
        elif 0 <= calibrate_angle < 30:
            view = "2" # Left
        elif 120 < calibrate_angle < 150:
            view = "-1" # Up-Right
        elif 150 <= calibrate_angle < 180:
            view = "-2" # Right

        if 45 <= calibrate_angle < 135:
            view2 = "0" # Up
        elif 0 <= calibrate_angle < 45:
            view2 = "1" # Left
        elif 135 <= calibrate_angle < 180:
            view2 = "-1" # Right
        

        # Calculate angle
                
        # Step 1 - side view shoulder position
        left_shoulder_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'side')
        right_shoulder_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'side')
        if 0 <= left_shoulder_angle < 20:
            left_shoulder_score = 1
        elif 20 <= left_shoulder_angle < 45:
            left_shoulder_score = 2     
        elif 45 <= left_shoulder_angle < 90:
            left_shoulder_score = 3
        elif 90 <= left_shoulder_angle:
            left_shoulder_score = 4     

        if 0 <= right_shoulder_angle < 20:
            right_shoulder_score = 1
        elif 20 <= right_shoulder_angle < 45:
            right_shoulder_score = 2     
        elif 45 <= right_shoulder_angle < 90:
            right_shoulder_score = 3
        elif 90 <= right_shoulder_angle:
            right_shoulder_score = 4                    
        # Addition - front view shoulder abducted
        left_shoulder_abduct_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'front')
        right_shoulder_abduct_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'front')
        if left_shoulder_abduct_angle > 45:
            left_shoulder_abduct_score = 1
        else:
            left_shoulder_abduct_score = 0
        if right_shoulder_abduct_angle > 45:
            right_shoulder_abduct_score = 1 
        else:
            right_shoulder_abduct_score = 0

        step1_left_score = left_shoulder_score + left_shoulder_abduct_score
        step1_right_score = right_shoulder_score + right_shoulder_abduct_score
        print("step1 left score = " + str(step1_left_score) + " and step1 right score = " + str(step1_right_score))
        #print("Right Upper angle " + str(right_shoulder_angle))
        #print("Right Upper angle = " + str(right_shoulder_score))


               # Step 2 - side view elbow position
        left_lower_angle = calculate_angle(Left_shoulder_Pos, Left_elbow_Pos, Left_wrist_Pos, 'side')
        right_lower_angle = calculate_angle(Right_shoulder_Pos, Right_elbow_Pos, Right_wrist_Pos, 'side')   
        if 90 <= left_lower_angle <= 150:
            left_lower_score = 1
        else:
            left_lower_score = 2
        if 90 <= right_lower_angle < 150:
            right_lower_score = 1
        else:
            right_lower_score = 2           
        # Addition - front&top views forearm across midline
        forearm_intersection_point_xz = find_intersection_point(Left_elbow_Pos, Left_wrist_Pos, Right_elbow_Pos, Right_wrist_Pos, 'top')
        forearm_intersection_point_xy = find_intersection_point(Left_elbow_Pos, Left_wrist_Pos, Right_elbow_Pos, Right_wrist_Pos, 'front')
        if forearm_intersection_point_xz:
            wrist_midline = 1
            # print("Intersection point:", forearm_intersection_point_xz)
        elif forearm_intersection_point_xy:
            wrist_midline = 1
            # print("Intersection point:", forearm_intersection_point_xy)
        else:
            wrist_midline = 0
            # print("Lines are parallel, no intersection point")

        step2_left_score = left_lower_score + wrist_midline
        step2_right_score = right_lower_score + wrist_midline
        print("step2 left score = " + str(step2_left_score) + " and step2 right score = " + str(step2_right_score))


        # Step 3 - side view wrist position
        left_wrist_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'side')
        right_wrist_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'side')
        # Addition - top view wrist deviation
        left_wrist_deviation_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'top')
        right_wrist_deviation_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'top')

        step3_left_score = 2
        step3_right_score = 2
        #print("leftwrist angle = " + str(left_wrist_angle))
        #print("rightwrist angle = " + str(right_wrist_angle))

        # Step 4 - front view wrist twist
        left_wrist_twist_angle = calculate_angle(Left_index_Pos, Left_wrist_Pos, Left_thumb_Pos, 'front')
        right_wrist_twist_angle = calculate_angle(Right_index_Pos, Right_wrist_Pos, Right_thumb_Pos, 'front')

        step4_left_score = 2
        step4_right_score = 2


        # Table A score


        # Step 9 - side view neck position
        neck_angle = findAngle(Left_shoulder_Pos[0], Left_shoulder_Pos[1], Left_ear_Pos[0], Left_ear_Pos[1])
        if 12.5 <= neck_angle < 24.5:
            neck_score = 1
        elif 25 <= neck_angle < 29.5:
            neck_score = 2         
        elif 30 <= neck_angle:
            neck_score = 3
        else:
            neck_score = 4
        # Addition bending
        neck_bent_angle = calculate_angle(Right_shoulder_Pos, Neck_Pos, Head_Pos, 'top')
        if 75 <= neck_bent_angle < 105:
            neck_bent_score = 0
        else:
            neck_bent_score = 1
        # Addition  rotation
        calibrate_neck_angle = calculate_angle(Left_ear_Pos, Right_ear_Pos, (Right_ear_Pos[0], Right_ear_Pos[1],Right_ear_Pos[2]-1), 'top')

        # step9_score = neck_score + neck_bent_score
        step9_score = neck_score
        print("Neck angle = " + str(neck_angle))
        print("Neck score = " + str(neck_score))


        # Step 10 - side view trunk position
        trunk_angle = calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'side')
        if 178.5 < trunk_angle <= 180:
            trunk_score = 1
        elif 168.5 < trunk_angle <= 178.5:
            trunk_score = 2
        elif 150 <= trunk_angle <= 168.5:
            trunk_score = 3
        elif 0 <= trunk_angle < 150:
            trunk_score = 3
        # Addition bending
        trunk_bent_angle = calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'front')
        if 180 >= trunk_bent_angle > 162.5 :
            trunk_bent_score = 0
        else:
            trunk_bent_score = 1
        
        # step10_score = trunk_score + trunk_bent_score
        step10_score = trunk_score
        print("truk score = " + str(step10_score))


        # Step 11 - side&front views legs position
        left_knee_angle = calculate_angle(Left_hip_Pos, Left_knee_Pos, Left_ankle_Pos, 'side')
        right_knee_angle = calculate_angle(Right_hip_Pos, Right_knee_Pos, Right_ankle_Pos, 'side')
        if abs(left_knee_angle - right_knee_angle) >= 10 or abs(Left_ankle_Pos[1] - Right_ankle_Pos[1]) > abs(Hip_Pos[1] - Neck_Pos[1])/2.4: # Unbalance
            legs_score = 2
        else: # Balance
            legs_score = 1

        step11_score = legs_score
        print("step11 leg score = " + str(step11_score))



        global Muscle
        global MuscleN
        global Calweight
        global Weight    

        Muscle = "No"
        Weight = 0
    
        if Muscle == "Yes":
            MuscleN = 1
        elif Muscle == "No":
            MuscleN = 0
        else:
            MuscleN = 0

        Calweight = 0
        if Weight < 1.99 :
            Calweight = 0
        elif 1.99 <= Weight < 9.97 and Muscle == "No":
            Calweight = 1
        elif 1.99 <= Weight < 9.97 and Muscle == "Yes":
            Calweight = 2 
        elif Weight > 9.97 and Muscle == "No":
            Calweight = 2
        elif Weight > 9.97 and Muscle == "Yes":
            Calweight = 3
        # print("Start")
        # print("")
        # print("Calweight = " + str(Calweight))

        LC, RC = find_rula_opp('TableA.csv','TableB.csv','TableC.csv')

        # cv2.imshow('Frame', frame)
        resized_frame = cv2.resize(output_image, (1980, 1080))
        image_placeholder.image(resized_frame, channels="RGB", width = 900, caption = "")    
        text_placeholder1.write("Left RULA grand score = " + str(LC))
        text_placeholder2.write("Right RULA grand score = " + str(RC))
                
        #plt.clf() 
        plt.pause(.25)

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Close MediaPipe pose model
    pose.close()

    # plt.show()
    st.pyplot()

def find_rula_opp(input1,input2,input3):
    tablea=pd.read_csv(str(input1))
    tableb=pd.read_csv(str(input2))
    tablec=pd.read_csv(str(input3))

    #Table LEFT A:
    col_name=str(step3_left_score)+'WT'+str(step4_left_score)
    LA=tablea[(tablea['UpperArm']==step1_left_score) & (tablea['LowerArm']==step2_left_score)]
    LA=LA[col_name].values[0]
    # print("LA = " + str(LA))

    #Table RIGHT A:
    col_name=str(step3_right_score)+'WT'+str(step4_right_score)
    RA=tablea[(tablea['UpperArm']==step1_right_score) & (tablea['LowerArm']==step2_right_score)]
    RA=RA[col_name].values[0]
    # print("RA = " + str(RA))

    #Table LEFT B:
    col_name=str(step10_score)+str(step11_score)
    LB=tableb[(tableb['Neck']==step9_score)]
    LB=LB[col_name].values[0]
    # print("LB = " + str(LB))

    #Table RIGHT B:
    RB = LB
    # print("RB = " + str(RB))

    CLA = Calweight + MuscleN + int(LA) 
    CRA = Calweight + MuscleN + int(RA)
    CLB = Calweight + MuscleN + int(LB)
    CRB = Calweight + MuscleN + int(RB)
    # print("CLA = " + str(CLA))
    # print("CLB = " + str(CLB))

    #Table LEFT C
    if CLA>=8:
        CLA=8
    if CLB>=7:
        CLB=7
    col_name=str(CLB)
    LC=tablec[(tablec['Score']==CLA)]
    LC=LC[col_name].values[0]
    # print("LC = " + str(LC))

    #Table RIGHT C
    if CRA>=8:
        CRA=8
    if CRB>=7:
        CRB=7
    col_name=str(CRB)
    RC=tablec[(tablec['Score']==CRA)]
    RC=RC[col_name].values[0]
    # print("RC = " + str(RC))

    return LC, RC  

def conditionMuscle2():
    global Muscle
    options1 = ['No', 'Yes']
    Muscle = st.selectbox("Is the posture mainly static or action repeated occurs?", options1)
    # st.write("Muscle = " + str(Muscle))
    return Muscle

def conditionWeight2():
    global Weight
    values = list(range(0, 16))
    Weight = st.select_slider("What is the weight of the load?", options = values)
    # st.write("Weight = " + str(Weight))
    return Weight

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
                        default=0.7)

    args = parser.parse_args()

    return args

def sendWarning(x):
    pass



video_pose_estimation2(0)