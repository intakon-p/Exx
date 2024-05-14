import mimetypes
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import tkinter as tk
import customtkinter as ctk
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math as m


mimetypes.init()

root = ctk.CTk()

variable1=StringVar()    
variable2=StringVar()    

root.geometry ("700x840")

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


Thumb_Tip = None
Index_Tip = None
Pinky_Tip = None
handswrist = None


# #Old def tkmain
# def calculateAngle(landmark1, landmark2, landmark3):
#     '''
#     This function calculates angle between three different landmarks.
#     Args:
#         landmark1: The first landmark containing the x,y and z coordinates.
#         landmark2: The second landmark containing the x,y and z coordinates.
#         landmark3: The third landmark containing the x,y and z coordinates.
#     Returns:
#         angle: The calculated angle between the three landmarks.

#     '''

#     # Get the required landmarks coordinates.
#     x1, y1, _ = landmark1
#     x2, y2, _ = landmark2
#     x3, y3, _ = landmark3

#     # Calculate the angle between the three points
#     angle = m.degrees(m.atan2(y3 - y2, x3 - x2) - m.atan2(y1 - y2, x1 - x2))
    
#     # Check if the angle is less than zero.
#     if angle < 0:

#         # Add 360 to the found angle.
#         angle += 360
    
#     # Return the calculated angle.
#     return angle

def calculateDistance(keypoint1, keypoint2):
    x1, y1, _ = keypoint1
    x2, y2, _ = keypoint2

    # Calculate the Distance between the two points
    dis = m.sqrt( ((x2 - x1)**2)+((y2 - y1))**2)

    
    # Return the calculated Distance.
    return dis



# #def calculate_angle1(keypoint1, keypoint2, keypoint3, plane, Tellside):
#     global angle
#     # Extract x, y, and z coordinates for each keypoint
#     x1, y1, z1 = keypoint1
#     x2, y2, z2 = keypoint2
#     x3, y3, z3 = keypoint3

#     # For test z-coordinates
#     keypoint1 = x1, y1, z1*101
#     keypoint2 = x2, y2, z2*101
#     keypoint3 = x3, y3, z3*101

#     if plane == 'front':
#         # Calculate vectors between keypoints in the XY-plane
#         vector1 = np.array([x1 - x2, y1 - y2, 0])
#         vector2 = np.array([x3 - x2, y3 - y2, 0])
#     elif plane == 'top':
#         # Calculate vectors between keypoints in the XZ-plane
#         vector1 = np.array([x1 - x2, 0, (z1 - z2)*101])
#         vector2 = np.array([x3 - x2, 0, (z3 - z2)*101])
#     elif plane == 'side':
#         # Calculate vectors between keypoints in the YZ-plane
#         vector1 = np.array([0, y1 - y2, (z1 - z2)*101])
#         vector2 = np.array([0, y3 - y2, (z3 - z2)*101])

#     # Calculate dot product and magnitudes
#     dot_product = np.dot(vector1, vector2)
#     magnitude_vector1 = np.linalg.norm(vector1)
#     magnitude_vector2 = np.linalg.norm(vector2)

#     # Calculate cosine of the angle
#     cos_theta = dot_product / (magnitude_vector1 * magnitude_vector2)

#     # Convert cosine to angle in degrees
#     angle = np.degrees(np.arccos(cos_theta))
    
#     # Determine the orientation of the body relative to the camera
#     if 70 <= angle <= 100:
#         Tellside = 0  # Body is facing the camera
#     else:
#         Tellside = 1  # Side body is facing the camera
    
#     return angle, Tellside





def calculate_angle(keypoint1, keypoint2, keypoint3, plane):
    global angle
    # Extract x, y, and z coordinates for each keypoint
    x1, y1, z1 = keypoint1
    x2, y2, z2 = keypoint2
    x3, y3, z3 = keypoint3

    # For test z-coordinates
    keypoint1 = x1, y1, z1*101
    keypoint2 = x2, y2, z2*101
    keypoint3 = x3, y3, z3*101

    if plane == 'front':
        # Calculate vectors between keypoints in the XY-plane
        vector1 = np.array([x1 - x2, y1 - y2, 0])
        vector2 = np.array([x3 - x2, y3 - y2, 0])
    elif plane == 'top':
        # Calculate vectors between keypoints in the XZ-plane
        vector1 = np.array([x1 - x2, 0, (z1 - z2)*101])
        vector2 = np.array([x3 - x2, 0, (z3 - z2)*101])
    elif plane == 'side':
        # Calculate vectors between keypoints in the YZ-plane
        vector1 = np.array([0, y1 - y2, (z1 - z2)*101])
        vector2 = np.array([0, y3 - y2, (z3 - z2)*101])

    # Calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    # Calculate cosine of the angle
    cos_theta = dot_product / (magnitude_vector1 * magnitude_vector2)

    # Convert cosine to angle in degrees
    angle = np.degrees(np.arccos(cos_theta))
    # print("Left Shoulder: " + str(angle))
    # print("elbow " + str(keypoint1))
    # print("shoulder " + str(keypoint2))
    # print("hip " + str(keypoint3))

    return angle

def find_intersection_point(keypoint1, keypoint2, keypoint3, keypoint4, plane):
    # Extract x, y, and z coordinates for each keypoint
    x1, y1, z1 = keypoint1
    x2, y2, z2 = keypoint2
    x3, y3, z3 = keypoint3
    x4, y4, z4 = keypoint4

    # For test z-coordinates
    keypoint1 = x1, y1, z1 * 101
    keypoint2 = x2, y2, z2 * 101
    keypoint3 = x3, y3, z3 * 101
    keypoint4 = x4, y4, z4 * 101
    
    if plane == 'front':
        # Calculate slopes of the lines
        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y4 - y3) / (x4 - x3)
        
        # Check if lines are parallel
        if slope1 == slope2:
            return False  # Lines are parallel, no intersection point
        
        # Calculate the intersection point
        x_intersect = (slope1 * x1 - slope2 * x3 + y3 - y1) / (slope1 - slope2)
        y_intersect = slope1 * (x_intersect - x1) + y1
        z_intersect = 0

        # Check if intersection point lies on both line segments
        if (min(x1, x2) <= x_intersect <= max(x1, x2) and
                min(y1, y2) <= y_intersect <= max(y1, y2) and
                min(x3, x4) <= x_intersect <= max(x3, x4) and
                min(y3, y4) <= y_intersect <= max(y3, y4)):
            return x_intersect, y_intersect, z_intersect
        else:
            return False  # Intersection point lies outside the line segments

    elif plane == 'top':
        # Calculate slopes of the lines
        slope1 = (z2 - z1) / (x2 - x1)
        slope2 = (z4 - z3) / (x4 - x3)
        
        # Check if lines are parallel
        if slope1 == slope2:
            return False  # Lines are parallel, no intersection point
        
        # Calculate the intersection point
        x_intersect = (slope1 * x1 - slope2 * x3 + z3 - z1) / (slope1 - slope2)
        y_intersect = 0
        z_intersect = slope1 * (x_intersect - x1) + z1

        # Check if intersection point lies on both line segments
        if (min(x1, x2) <= x_intersect <= max(x1, x2) and
                min(z1, z2) <= z_intersect <= max(z1, z2) and
                min(x3, x4) <= x_intersect <= max(x3, x4) and
                min(z3, z4) <= z_intersect <= max(z3, z4)):
            return x_intersect, y_intersect, z_intersect
        else:
            return False  # Intersection point lies outside the line segments

    elif plane == 'side':
        # Calculate slopes of the lines
        slope1 = (y2 - y1) / (z2 - z1)
        slope2 = (y4 - y3) / (z4 - z3)
        
        # Check if lines are parallel
        if slope1 == slope2:
            return False  # Lines are parallel, no intersection point
        
        # Calculate the intersection point
        x_intersect = 0
        y_intersect = (slope1 * y1 - slope2 * y3 + z3 - z1) / (slope1 - slope2)
        z_intersect = slope1 * (y_intersect - y1) + z1

        # Check if intersection point lies on both line segments
        if (min(y1, y2) <= y_intersect <= max(y1, y2) and
                min(z1, z2) <= z_intersect <= max(z1, z2) and
                min(y3, y4) <= y_intersect <= max(y3, y4) and
                min(z3, z4) <= z_intersect <= max(z3, z4)):
            return x_intersect, y_intersect, z_intersect
        else:
            return False  # Intersection point lies outside the line segments

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    if y1 == 0:
        return float('inf')  # Return infinity if y1 is zero to avoid division by zero
    else:
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / m.pi) * theta
        return degree


#for webcam
def webcam2(name):
    # Initialize MediaPipe pose model

    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

    
    


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
        (26, 28),   # Right knee to right ankle

        (11, 12),  # Left shoulder to right shoulder

        (8, 6),  
        (6, 5),   
        (5, 4),  
        (4, 0),    
        (0, 1),    
        (1, 2),    
        (2, 3),  
        (3, 7),
        (10, 9),
        (15,19),
        (16,20)
    ]


    

    # Open the webcam
    cap = cv2.VideoCapture(name)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

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
    

    global Index_Tip
    global handswrist
    global Thumb_CMC
    global Thumb_MCP
    global Thumb_IP
    global Thumb_Tip
    global Index_MCP
    global Index_PIP
    global Index_DIP
    global Middle_MCP
    global Middle_PIP
    global Middle_DIP
    global Middle_Tip
    global Ring_MCP
    global Ring_PIP
    global Ring_DIP
    global Ring_Tip
    global Pinky_MCP
    global Pinky_PIP
    global Pinky_DIP
    global Pinky_Tip

    

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
    
    # #trymultitreading
    # def pose_estimation_thread(name):
    #     video_pose_estimation2(name)
    
    # # Start the pose estimation thread
    # pose_thread = threading.Thread(target=pose_estimation_thread, args=(0,))
    # pose_thread.start()
    



    while True:
        #Resetค่า##############################################3
        Pinky_Tip=None
        Thumb_Tip=None
        ########################################################3 
        ret, frame = cap.read()
        if not ret:
            break
   
        # Detect keypoints
        pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        
        
        # If pose detected, plot keypoints and lines
       

        if pose_results.pose_landmarks :
            image_with_keypoints = frame.copy()  # Create a copy of the captured image
            keypoints_3d = []
            #mp_drawing.draw_landmarks(image_with_keypoints,pose_results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            
        
            for landmark in pose_results.pose_landmarks.landmark:
                cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z

                # Store keypoints
                keypoints_3d.append((cx, cy, cz))

            Nose_Pos = keypoints_3d[0]
            Left_eye_inner_Pos = keypoints_3d[1]
            Left_eye_Pos = keypoints_3d[2]
            Left_eye_outer_Pos = keypoints_3d[3]
            Right_eye_inner_Pos = keypoints_3d[4]
            Right_eye_Pos = keypoints_3d[5]
            Right_eye_outer_Pos = keypoints_3d[6]
            Left_ear_Pos = keypoints_3d[7]
            Right_ear_Pos = keypoints_3d[8]
            Mouth_left_Pos = keypoints_3d[9]
            Mouth_right_Pos = keypoints_3d[10]
            Left_shoulder_Pos = keypoints_3d[11]
            Right_shoulder_Pos = keypoints_3d[12]
            Left_elbow_Pos = keypoints_3d[13]
            Right_elbow_Pos = keypoints_3d[14]
            Left_wrist_Pos = keypoints_3d[15]
            Right_wrist_Pos = keypoints_3d[16]
            Left_pinky_Pos = keypoints_3d[17]
            Right_pinky_Pos = keypoints_3d[18]
            Left_index_Pos = keypoints_3d[19]
            Right_index_Pos = keypoints_3d[20]
            Left_thumb_Pos = keypoints_3d[21]
            Right_thumb_Pos = keypoints_3d[22]
            Left_hip_Pos = keypoints_3d[23]
            Right_hip_Pos = keypoints_3d[24]
            Left_knee_Pos = keypoints_3d[25]
            Right_knee_Pos = keypoints_3d[26]
            Left_ankle_Pos = keypoints_3d[27]
            Right_ankle_Pos = keypoints_3d[28]
            Left_heel_Pos = keypoints_3d[29]
            Right_heel_Pos = keypoints_3d[30]
            Left_foot_index_Pos = keypoints_3d[31]
            Right_foot_index_Pos = keypoints_3d[32]

            # Calculate additional positions
            Neck_Pos = ((Right_shoulder_Pos[0] + Left_shoulder_Pos[0]) / 2,
                        (Right_shoulder_Pos[1] + Left_shoulder_Pos[1]) / 2,
                        (Right_shoulder_Pos[2] + Left_shoulder_Pos[2]) / 2)

            Head_Pos = ((Right_ear_Pos[0] + Left_ear_Pos[0]) / 2,
                        (Right_ear_Pos[1] + Left_ear_Pos[1]) / 2,
                        (Right_ear_Pos[2] + Left_ear_Pos[2]) / 2)

            Hip_Pos = ((Right_hip_Pos[0] + Left_hip_Pos[0]) / 2,
                    (Right_hip_Pos[1] + Left_hip_Pos[1]) / 2,
                    (Right_hip_Pos[2] + Left_hip_Pos[2]) / 2)
            
            # Separate 3D coordinates into individual lists
            keypoints_x, keypoints_y, keypoints_z = zip(*keypoints_3d)

            # Draw lines between keypoints
            for connection in keypoint_connections:
                start_point = connection[0]
                end_point = connection[1]
                start_coords = (int(keypoints_3d[start_point][0]), int(keypoints_3d[start_point][1]))
                end_coords = (int(keypoints_3d[end_point][0]), int(keypoints_3d[end_point][1]))
                cv2.line(image_with_keypoints, start_coords, end_coords, color=(255, 0, 0), thickness=2)  # Red line
            
                #forhand
            if hands_results.multi_hand_landmarks:
                handskeypoints_3d = []
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image_with_keypoints,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                     mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()) 
                    for landmark in hand_landmarks.landmark:
                        cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z
                        handskeypoints_3d.append((cx, cy, cz))
                    
                    handswrist = handskeypoints_3d[0]
                    Thumb_CMC = handskeypoints_3d[1]
                    Thumb_MCP = handskeypoints_3d[2]
                    Thumb_IP = handskeypoints_3d[3]
                    Thumb_Tip = handskeypoints_3d[4]
                    Index_MCP = handskeypoints_3d[5]
                    Index_PIP = handskeypoints_3d[6]
                    Index_DIP = handskeypoints_3d[7]
                    Index_Tip = handskeypoints_3d[8]
                    Middle_MCP = handskeypoints_3d[9]
                    Middle_PIP = handskeypoints_3d[10]
                    Middle_DIP = handskeypoints_3d[11]
                    Middle_Tip = handskeypoints_3d[12]
                    Ring_MCP = handskeypoints_3d[13]
                    Ring_PIP = handskeypoints_3d[14]
                    Ring_DIP = handskeypoints_3d[15]
                    Ring_Tip = handskeypoints_3d[16]
                    Pinky_MCP = handskeypoints_3d[17]
                    Pinky_PIP = handskeypoints_3d[18]
                    Pinky_DIP = handskeypoints_3d[19]
                    Pinky_Tip = handskeypoints_3d[20]

            # print("hand  index   = " + str(Index_Tip))
            # print("pose  index  = " + str(Right_index_Pos))
            # Index = handskeypoints_3d[8]
            # Index_text = "Index Tip"
            # Index_text_pos = (int(Index[0]), int(Index[1]))
            # cv2.putText(image_with_keypoints, Index_text, Index_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # thumb = handskeypoints_3d[4]
            # thumb_text = "Thumb Tip"
            # thumb_text_pos = (int(thumb[0]), int(thumb[1]))
            # cv2.putText(image_with_keypoints, thumb_text, thumb_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

            text_posx = 20
            text_step = 40

            # Display the text on the image
            #cv2.putText(image_with_keypoints, text, text_position, cv2.FONT_HERSHEY_PLAIN, font_scale, font_color, font_thickness)

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

            

            if 45 <= calibrate_angle < 120:
                view2 = 0 # Up
            elif 0 <= calibrate_angle < 45:
                view2 = 1 # Left
            elif 120 <= calibrate_angle < 180:
                view2 = -1 # Right

            #Tellside step test
            bodysideangle=calculate_angle(Left_shoulder_Pos, Right_shoulder_Pos, (Right_shoulder_Pos[0], Right_shoulder_Pos[1],Right_shoulder_Pos[2]-1), 'top')
            if 65 <=bodysideangle<= 110:
                #หันเข้ากล้อง
                Tellside = 0
            else :
                #หันข้าง
                Tellside = 1

            #cv2.putText(image_with_keypoints, "cali: " + str("{:0.2f}".format(calibrate_angle)), (10, text_posx*5), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #cv2.putText(image_with_keypoints, "view : " + str("{:0.2f}".format(view2)), (10, text_posx*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            
            
            cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(bodysideangle)), (10, text_posx*5), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(Tellside)), (10, text_posx*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                    
            # Step 1 - side view shoulder position
            #หันหน้า
            #if view2 == 0 :
            if Tellside == 0: 
              
                #step1
                left_shoulder_angle= calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'side')
                right_shoulder_angle= calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'side')
                left_shoulder_abduct_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'front')
                right_shoulder_abduct_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'front')
                #step2
                left_elbow_angle = calculate_angle(Left_shoulder_Pos, Left_elbow_Pos,Left_wrist_Pos, 'side')
                right_elbow_angle = calculate_angle(Right_shoulder_Pos, Right_elbow_Pos, Right_wrist_Pos, 'side')

                if  right_shoulder_abduct_angle > 90 :
                    right_shoulder_angle=10

                if left_shoulder_abduct_angle> 90 :
                    left_shoulder_angle=10
                #step 3
                #step3 (Accu
                left_wrist_angle = 180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'side')
                right_wrist_angle = 180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'side')

                #พึ่งเพิ่ม step3 กัน error
                if Pinky_Tip == None :
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'side'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'side'))
                if Pinky_Tip != None:
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Pinky_Tip, 'side'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos,Pinky_Tip, 'side'))
               

            #หันข้าง
            #if view2 == 1 :
            if Tellside == 1:
                #step1
                left_shoulder_angle= calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'front')
                right_shoulder_angle= calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'front')
                left_shoulder_abduct_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'side')
                right_shoulder_abduct_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'side')
                #step2
                left_elbow_angle = 180-calculate_angle(Left_shoulder_Pos, Left_elbow_Pos, Left_wrist_Pos, 'front')
                right_elbow_angle =180-calculate_angle(Right_shoulder_Pos, Right_elbow_Pos, Right_wrist_Pos, 'front')
                

                
                
                if  right_shoulder_angle > 50 :
                    right_shoulder_abduct_angle=10

                if left_shoulder_angle> 50 :
                    left_shoulder_abduct_angle=10

                #step3 (Accu75)
                #จับมือไม่ได้
                if Pinky_Tip == None :
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'front'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'front'))
                    
                    #step4วันนี้
                    step4_angleright=None
                    step4_angleleft=None
                    step4_left_score = 2
                    step4_right_score =2

                #จับมือได้
                if Pinky_Tip != None:
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Pinky_Tip, 'front'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos,Pinky_Tip, 'front'))
                    
                    
                    step4_angleleft = calculate_angle(Left_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                    if step4_angleleft < 90 :
                     step4_left_score=  1
                    else :
                     step4_left_score= 2
                    
                    step4_angleright = calculate_angle(Right_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                    #step4วันนี้
                    if step4_angleright < 90 :
                     step4_right_score=  1
                    else :
                     step4_right_score= 2
                    
                    

                print("step4ang right = "+str(step4_angleright))
                print("step4ang left = "+str(step4_angleleft))
                print("step4scoretestright =  "+ str(step4_right_score))
                print("step4scoretestleft =  "+ str(step4_left_score))
                cv2.putText(image_with_keypoints, "step4testleft: " + str("{:0.2f}".format(step4_left_score)), (10, text_posx+text_step*9), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                cv2.putText(image_with_keypoints, "step4testright : " + str("{:0.2f}".format(step4_right_score)), (10, text_posx+text_step*10), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                
                cv2.putText(image_with_keypoints, "L_wrist_angle : " + str("{:0.2f}".format(left_wrist_angle)), (10, text_posx+text_step*7), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                cv2.putText(image_with_keypoints, "R_wrist_angle : " + str("{:0.2f}".format(right_wrist_angle)), (10, text_posx+text_step*8), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
              
            #   if Left_index_Pos is None:
            #         print("Left hand index not detected")
            #         # If it's None, display "Cant detect"
            #         cv2.putText(image_with_keypoints, "Left hand can't detect" , (10, text_posx+text_step*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #   else:
            #         print("Left hand index detected")
            #         # If it's not None, display "Can detect"
            #         cv2.putText(image_with_keypoints, "can detect" , (10, text_posx+text_step*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                                    

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
            # print("step1 left score = " + str(step1_left_score) + " and step1 right score = " + str(step1_right_score))
            
            
            
            
            
            
            #cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(Tellside)), (10, text_posx), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            
            
            cv2.putText(image_with_keypoints, "L_upper_angle : " + str("{:0.2f}".format(left_shoulder_angle)), (10, text_posx), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #cv2.putText(image_with_keypoints, "L_upper_arm_score : " + str("{:0.2f}".format(left_shoulder_score)), (300, text_posx),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            cv2.putText(image_with_keypoints, "R_upper_angle : " + str("{:0.2f}".format(right_shoulder_angle)), (10, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            #cv2.putText(image_with_keypoints, "R_upper_score : " + str("{:0.2f}".format(right_shoulder_score)), (300, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            
            cv2.putText(image_with_keypoints, "L_upper_abduct angle : " + str("{:0.2f}".format(left_shoulder_abduct_angle)), (300, text_posx),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            cv2.putText(image_with_keypoints, "R_upper_abduct angle : " + str("{:0.2f}".format(right_shoulder_abduct_angle)), (300, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            
            # Step 2 - side view elbow position
               
          
            if 60<= left_elbow_angle <= 105:
                left_elbow_score = 1
            else:
                left_elbow_score = 2
            if 60 <= right_elbow_angle < 105:
                right_elbow_score = 1
            else:
                right_elbow_score = 2                
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

            step2_left_score = left_elbow_score + wrist_midline
            step2_right_score = right_elbow_score + wrist_midline
            #print("step2 left score = " + str(step2_left_score) + " and step2 right score = " + str(step2_right_score))


            # Step 3 - side view wrist position
            if 0 <=left_wrist_angle <=5:
                left_wrist_score=1
            elif 5<left_wrist_angle<=15:
                left_wrist_score=2
            elif 15<=left_wrist_angle:
                left_wrist_score=3
            
            if 0 <=right_wrist_angle <=5:
                right_wrist_score=1
            elif 5<right_wrist_angle<=15:
                right_wrist_score=2
            elif 15<=right_wrist_angle:
                right_wrist_score=3

            # Addition - top view wrist deviation
            left_wrist_deviation_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'top')
            right_wrist_deviation_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'top')

            step3_left_score = left_wrist_score
            step3_right_score = right_wrist_score
            cv2.putText(image_with_keypoints, "L_wrist_score : " + str("{:0.2f}".format(left_wrist_score)), (600, text_posx+text_step*7), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "R_wrist_score : " + str("{:0.2f}".format(right_wrist_score)), (600, text_posx+text_step*8), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            
            # print("leftwrist angle = " + str(left_wrist_angle))
            # print("rightwrist angle = " + str(right_wrist_angle))

            # Step 4 - front view wrist twist
            # left_wrist_twist_angle = calculate_angle(Left_index_Pos, Left_wrist_Pos, Left_thumb_Pos, 'front')
            # right_wrist_twist_angle = calculate_angle(Right_index_Pos, Right_wrist_Pos, Right_thumb_Pos, 'front')

            step4_left_score = 2
            step4_right_score = 2


            # Table A score


            # Step 9 - side view neck position
            neck_angle = findAngle(Neck_Pos[0], Neck_Pos[1], Nose_Pos[0], Nose_Pos[1]) - 30
            # neck_angle = calculate_angle((Neck_Pos[0], Neck_Pos[1], Neck_Pos[2] - 1), Neck_Pos, Nose_Pos) - 40
            print("neck_ = " + str(neck_angle))
            if 0 <= neck_angle < 10.5:
                neck_score = 1
            elif 10.5 <= neck_angle < 20.5:
                neck_score = 2         
            elif 20.5 <= neck_angle:
                neck_score = 3
            else:
                neck_score = 4
            print("neck score = " + str(neck_score))
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
            #print("step9 neck score = " + str(step9_score))


            #step 10
            #print ("Cali AN = "+str(calibrate_angle))
            #print("view2 = " + str(view2))
            if view2 == -1: #หันขวา
                trunk_angle =180-calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'front')
    
            if view2 == 1: #หันซ้าย
                trunk_angle=180 - calculate_angle(Left_knee_Pos, Left_hip_Pos, Left_shoulder_Pos, 'front')
            

            if view2 != 0 :
                if 0 < trunk_angle <= 8.3:
                        trunk_score = 1
                elif 8.3 < trunk_angle <= 20:
                        trunk_score = 2
                elif 20 <= trunk_angle <= 60:
                        trunk_score = 3
                elif 60 < trunk_angle:
                        trunk_score = 4
                
            if view2 == 0:
                trunk_angle=0
                trunk_score=1
            # Addition bending
            # trunk_bent_angle = calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'front')
            # if 180 >= trunk_bent_angle > 162.5 :
            #     trunk_bent_score = 0
            # else:
            #     trunk_bent_score = 1
            # step10_score = trunk_score + trunk_bent_score
            step10_score = trunk_score
            #print("step10 trunk angle = " + str(trunk_angle))
            #print("step10 trunk score = " + str(step10_score))

            # Step 11 - side&front views legs position
            left_knee_angle = calculate_angle(Left_hip_Pos, Left_knee_Pos, Left_ankle_Pos, 'side')
            right_knee_angle = calculate_angle(Right_hip_Pos, Right_knee_Pos, Right_ankle_Pos, 'side')
            if abs(left_knee_angle - right_knee_angle) >= 10 or abs(Left_ankle_Pos[1] - Right_ankle_Pos[1]) > abs(Hip_Pos[1] - Neck_Pos[1])/2.4: # Unbalance
                legs_score = 2
            else: # Balance
                legs_score = 1

            step11_score = legs_score
            #print("step11 leg score = " + str(step11_score))

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
            

            cv2.imshow("Image with Keypoints", image_with_keypoints)  

        else:
            cv2.imshow("Image with Keypoints", frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        print("Left RULA grand score = " + str(LC))
        print("Right RULA grand score = " + str(RC))
        variable1.set("LC : " + str(LC))
        variable2.set("RC : " + str(RC))
        root.update()
        
            
    

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Close MediaPipe pose model
    pose.close()

#for webcam
def video_pose_estimation(name):
    # Initialize MediaPipe pose model

    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

    
    
    


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
        (26, 28),   # Right knee to right ankle

        (11, 12),  # Left shoulder to right shoulder

        (8, 6),  
        (6, 5),   
        (5, 4),  
        (4, 0),    
        (0, 1),    
        (1, 2),    
        (2, 3),  
        (3, 7),
        (10, 9),
        (15,19),
        (16,20)
    ]


    

    # Open the webcam
    cap = cv2.VideoCapture(name)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
   #Test
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter('output_video.mp4', fourcc,fps, (frame_width, frame_height))
    # List to store frames
    frames = []

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
    

    global Index_Tip
    global handswrist
    global Thumb_CMC
    global Thumb_MCP
    global Thumb_IP
    global Thumb_Tip
    global Index_MCP
    global Index_PIP
    global Index_DIP
    global Middle_MCP
    global Middle_PIP
    global Middle_DIP
    global Middle_Tip
    global Ring_MCP
    global Ring_PIP
    global Ring_DIP
    global Ring_Tip
    global Pinky_MCP
    global Pinky_PIP
    global Pinky_DIP
    global Pinky_Tip

    

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
    
    # #trymultitreading
    # def pose_estimation_thread(name):
    #     video_pose_estimation2(name)
    
    # # Start the pose estimation thread
    # pose_thread = threading.Thread(target=pose_estimation_thread, args=(0,))
    # pose_thread.start()
    
    ###########try skip frame
    count=0

    while True:
        ############Reset
        Pinky_Tip=None
        Thumb_Tip=None 

        #################
        ret, frame = cap.read()
        ###########try skip frame
        # if ret:
        #     count += 20 # i.e. at 30 fps, this advances one second
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        


    
        ############
        
        if not ret:
            break
   
        # Detect keypoints
        pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        
        
        # If pose detected, plot keypoints and lines
       

        if pose_results.pose_landmarks :
            image_with_keypoints = frame.copy()  # Create a copy of the captured image
            keypoints_3d = []
            #mp_drawing.draw_landmarks(image_with_keypoints,pose_results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            
        
            for landmark in pose_results.pose_landmarks.landmark:
                cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z

                # Store keypoints
                keypoints_3d.append((cx, cy, cz))

            Nose_Pos = keypoints_3d[0]
            Left_eye_inner_Pos = keypoints_3d[1]
            Left_eye_Pos = keypoints_3d[2]
            Left_eye_outer_Pos = keypoints_3d[3]
            Right_eye_inner_Pos = keypoints_3d[4]
            Right_eye_Pos = keypoints_3d[5]
            Right_eye_outer_Pos = keypoints_3d[6]
            Left_ear_Pos = keypoints_3d[7]
            Right_ear_Pos = keypoints_3d[8]
            Mouth_left_Pos = keypoints_3d[9]
            Mouth_right_Pos = keypoints_3d[10]
            Left_shoulder_Pos = keypoints_3d[11]
            Right_shoulder_Pos = keypoints_3d[12]
            Left_elbow_Pos = keypoints_3d[13]
            Right_elbow_Pos = keypoints_3d[14]
            Left_wrist_Pos = keypoints_3d[15]
            Right_wrist_Pos = keypoints_3d[16]
            Left_pinky_Pos = keypoints_3d[17]
            Right_pinky_Pos = keypoints_3d[18]
            Left_index_Pos = keypoints_3d[19]
            Right_index_Pos = keypoints_3d[20]
            Left_thumb_Pos = keypoints_3d[21]
            Right_thumb_Pos = keypoints_3d[22]
            Left_hip_Pos = keypoints_3d[23]
            Right_hip_Pos = keypoints_3d[24]
            Left_knee_Pos = keypoints_3d[25]
            Right_knee_Pos = keypoints_3d[26]
            Left_ankle_Pos = keypoints_3d[27]
            Right_ankle_Pos = keypoints_3d[28]
            Left_heel_Pos = keypoints_3d[29]
            Right_heel_Pos = keypoints_3d[30]
            Left_foot_index_Pos = keypoints_3d[31]
            Right_foot_index_Pos = keypoints_3d[32]

            # Calculate additional positions
            Neck_Pos = ((Right_shoulder_Pos[0] + Left_shoulder_Pos[0]) / 2,
                        (Right_shoulder_Pos[1] + Left_shoulder_Pos[1]) / 2,
                        (Right_shoulder_Pos[2] + Left_shoulder_Pos[2]) / 2)

            Head_Pos = ((Right_ear_Pos[0] + Left_ear_Pos[0]) / 2,
                        (Right_ear_Pos[1] + Left_ear_Pos[1]) / 2,
                        (Right_ear_Pos[2] + Left_ear_Pos[2]) / 2)

            Hip_Pos = ((Right_hip_Pos[0] + Left_hip_Pos[0]) / 2,
                    (Right_hip_Pos[1] + Left_hip_Pos[1]) / 2,
                    (Right_hip_Pos[2] + Left_hip_Pos[2]) / 2)
            
            # Separate 3D coordinates into individual lists
            keypoints_x, keypoints_y, keypoints_z = zip(*keypoints_3d)

            # Draw lines between keypoints
            for connection in keypoint_connections:
                start_point = connection[0]
                end_point = connection[1]
                start_coords = (int(keypoints_3d[start_point][0]), int(keypoints_3d[start_point][1]))
                end_coords = (int(keypoints_3d[end_point][0]), int(keypoints_3d[end_point][1]))
                cv2.line(image_with_keypoints, start_coords, end_coords, color=(255, 0, 0), thickness=2)  # Red line
            
                #forhand
            if hands_results.multi_hand_landmarks:
                handskeypoints_3d = []
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image_with_keypoints,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()) 
                    for landmark in hand_landmarks.landmark:
                        cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z
                        handskeypoints_3d.append((cx, cy, cz))
                    
                    handswrist = handskeypoints_3d[0]
                    Thumb_CMC = handskeypoints_3d[1]
                    Thumb_MCP = handskeypoints_3d[2]
                    Thumb_IP = handskeypoints_3d[3]
                    Thumb_Tip = handskeypoints_3d[4]
                    Index_MCP = handskeypoints_3d[5]
                    Index_PIP = handskeypoints_3d[6]
                    Index_DIP = handskeypoints_3d[7]
                    Index_Tip = handskeypoints_3d[8]
                    Middle_MCP = handskeypoints_3d[9]
                    Middle_PIP = handskeypoints_3d[10]
                    Middle_DIP = handskeypoints_3d[11]
                    Middle_Tip = handskeypoints_3d[12]
                    Ring_MCP = handskeypoints_3d[13]
                    Ring_PIP = handskeypoints_3d[14]
                    Ring_DIP = handskeypoints_3d[15]
                    Ring_Tip = handskeypoints_3d[16]
                    Pinky_MCP = handskeypoints_3d[17]
                    Pinky_PIP = handskeypoints_3d[18]
                    Pinky_DIP = handskeypoints_3d[19]
                    Pinky_Tip = handskeypoints_3d[20]

            # print("hand  index   = " + str(Index_Tip))
            # print("pose  index  = " + str(Right_index_Pos))
            # Index = handskeypoints_3d[8]
            # Index_text = "Index Tip"
            # Index_text_pos = (int(Index[0]), int(Index[1]))
            # cv2.putText(image_with_keypoints, Index_text, Index_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # thumb = handskeypoints_3d[4]
            # thumb_text = "Thumb Tip"
            # thumb_text_pos = (int(thumb[0]), int(thumb[1]))
            # cv2.putText(image_with_keypoints, thumb_text, thumb_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            
            
            

            text_posx = 20
            text_step = 40

            # Display the text on the image
            #cv2.putText(image_with_keypoints, text, text_position, cv2.FONT_HERSHEY_PLAIN, font_scale, font_color, font_thickness)

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

            

            if 45 <= calibrate_angle < 120:
                view2 = 0 # Up
            elif 0 <= calibrate_angle < 45:
                view2 = 1 # Left
            elif 120 <= calibrate_angle < 180:
                view2 = -1 # Right

            #Tellside step test
            bodysideangle=calculate_angle(Left_shoulder_Pos, Right_shoulder_Pos, (Right_shoulder_Pos[0], Right_shoulder_Pos[1],Right_shoulder_Pos[2]-1), 'top')
            if 65 <=bodysideangle<= 110:
                #หันเข้ากล้อง
                Tellside = 0
            else :
                #หันข้าง
                Tellside = 1

            #cv2.putText(image_with_keypoints, "cali: " + str("{:0.2f}".format(calibrate_angle)), (10, text_posx*5), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #cv2.putText(image_with_keypoints, "view : " + str("{:0.2f}".format(view2)), (10, text_posx*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            
            
            cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(bodysideangle)), (10, text_posx*5), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(Tellside)), (10, text_posx*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                    
            # Step 1 - side view shoulder position
            #หันหน้า
            #if view2 == 0 :
            if Tellside == 0: 
              
              #step1
              left_shoulder_angle= calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'side')
              right_shoulder_angle= calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'side')
              left_shoulder_abduct_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'front')
              right_shoulder_abduct_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'front')
              #step2
              left_elbow_angle = calculate_angle(Left_shoulder_Pos, Left_elbow_Pos,Left_wrist_Pos, 'side')
              right_elbow_angle = calculate_angle(Right_shoulder_Pos, Right_elbow_Pos, Right_wrist_Pos, 'side')

              if  right_shoulder_abduct_angle > 90 :
                  right_shoulder_angle=10

              if left_shoulder_abduct_angle> 90 :
                  left_shoulder_angle=10
               #step 3
               #step3 ยัง
              #step3 (Accu75)
              if Pinky_Tip == None :
                  left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'front'))
                  right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'front'))
              if Pinky_Tip != None:
                  left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Pinky_Tip, 'front'))
                  right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos,Pinky_Tip, 'front'))
              
              cv2.putText(image_with_keypoints, "L_wrist_angle : " + str("{:0.2f}".format(left_wrist_angle)), (10, text_posx+text_step*7), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
              cv2.putText(image_with_keypoints, "R_wrist_angle : " + str("{:0.2f}".format(right_wrist_angle)), (10, text_posx+text_step*8), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
               

            #หันข้าง
            #if view2 == 1 :
            if Tellside == 1:
              #step1
              left_shoulder_angle= calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'front')
              right_shoulder_angle= calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'front')
              left_shoulder_abduct_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'side')
              right_shoulder_abduct_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'side')
              #step2
              left_elbow_angle = 180-calculate_angle(Left_shoulder_Pos, Left_elbow_Pos, Left_wrist_Pos, 'front')
              right_elbow_angle =180-calculate_angle(Right_shoulder_Pos, Right_elbow_Pos, Right_wrist_Pos, 'front')
              

              
              
              if  right_shoulder_angle > 50 :
                  right_shoulder_abduct_angle=10

              if left_shoulder_angle> 50 :
                  left_shoulder_abduct_angle=10

              #step3 (Accu75)
                              #step3 (Accu75)
                #จับมือไม่ได้
              if Pinky_Tip == None :
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'front'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'front'))
                    
                    #step4วันนี้
                    step4_angleright=None
                    step4_angleleft=None
                    step4_left_score = 2
                    step4_right_score =2

                #จับมือได้
              if Pinky_Tip != None:
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Pinky_Tip, 'front'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos,Pinky_Tip, 'front'))
                    
                    
                    step4_angleleft = calculate_angle(Left_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                    if step4_angleleft < 90 :
                     step4_left_score=  1
                    else :
                     step4_left_score= 2
                    
                    step4_angleright = calculate_angle(Right_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                    #step4วันนี้
                    if step4_angleright < 90 :
                     step4_right_score=  1
                    else :
                     step4_right_score= 2
                    
                    

              print("step4ang right = "+str(step4_angleright))
              print("step4ang left = "+str(step4_angleleft))
              print("step4scoretestright =  "+ str(step4_right_score))
              print("step4scoretestleft =  "+ str(step4_left_score))
              cv2.putText(image_with_keypoints, "step4testleft: " + str("{:0.2f}".format(step4_left_score)), (10, text_posx+text_step*9), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
              cv2.putText(image_with_keypoints, "step4testright : " + str("{:0.2f}".format(step4_right_score)), (10, text_posx+text_step*10), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
              
              
              cv2.putText(image_with_keypoints, "L_wrist_angle : " + str("{:0.2f}".format(left_wrist_angle)), (10, text_posx+text_step*7), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
              cv2.putText(image_with_keypoints, "R_wrist_angle : " + str("{:0.2f}".format(right_wrist_angle)), (10, text_posx+text_step*8), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
              
            #   if Left_index_Pos is None:
            #         print("Left hand index not detected")
            #         # If it's None, display "Cant detect"
            #         cv2.putText(image_with_keypoints, "Left hand can't detect" , (10, text_posx+text_step*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #   else:
            #         print("Left hand index detected")
            #         # If it's not None, display "Can detect"
            #         cv2.putText(image_with_keypoints, "can detect" , (10, text_posx+text_step*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                                    

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
            # print("step1 left score = " + str(step1_left_score) + " and step1 right score = " + str(step1_right_score))
            
            
            
            
            
            
            #cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(Tellside)), (10, text_posx), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            
            
            cv2.putText(image_with_keypoints, "L_upper_angle : " + str("{:0.2f}".format(left_shoulder_angle)), (10, text_posx), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #cv2.putText(image_with_keypoints, "L_upper_arm_score : " + str("{:0.2f}".format(left_shoulder_score)), (300, text_posx),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            cv2.putText(image_with_keypoints, "R_upper_angle : " + str("{:0.2f}".format(right_shoulder_angle)), (10, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            #cv2.putText(image_with_keypoints, "R_upper_score : " + str("{:0.2f}".format(right_shoulder_score)), (300, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            
            cv2.putText(image_with_keypoints, "L_upper_abduct angle : " + str("{:0.2f}".format(left_shoulder_abduct_angle)), (300, text_posx),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            cv2.putText(image_with_keypoints, "R_upper_abduct angle : " + str("{:0.2f}".format(right_shoulder_abduct_angle)), (300, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            
            # Step 2 - side view elbow position
               
          
            if 60<= left_elbow_angle <= 105:
                left_elbow_score = 1
            else:
                left_elbow_score = 2
            if 60 <= right_elbow_angle < 105:
                right_elbow_score = 1
            else:
                right_elbow_score = 2                 
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

            step2_left_score = left_elbow_score + wrist_midline
            step2_right_score = right_elbow_score + wrist_midline
            #print("step2 left score = " + str(step2_left_score) + " and step2 right score = " + str(step2_right_score))


            # Step 3 - side view wrist position
            if 0 <=left_wrist_angle <=5:
                left_wrist_score=1
            elif 5<left_wrist_angle<=15:
                left_wrist_score=2
            elif 15<=left_wrist_angle:
                left_wrist_score=3
            
            if 0 <=right_wrist_angle <=5:
                right_wrist_score=1
            elif 5<right_wrist_angle<=15:
                right_wrist_score=2
            elif 15<=right_wrist_angle:
                right_wrist_score=3

            # Addition - top view wrist deviation
            left_wrist_deviation_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'top')
            right_wrist_deviation_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'top')

            step3_left_score = left_wrist_score
            step3_right_score = right_wrist_score
            cv2.putText(image_with_keypoints, "L_wrist_score : " + str("{:0.2f}".format(left_wrist_score)), (600, text_posx+text_step*7), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "R_wrist_score : " + str("{:0.2f}".format(right_wrist_score)), (600, text_posx+text_step*8), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            
            # print("leftwrist angle = " + str(left_wrist_angle))
            # print("rightwrist angle = " + str(right_wrist_angle))

            # Step 4 - front view wrist twist
            # left_wrist_twist_angle = calculate_angle(Left_index_Pos, Left_wrist_Pos, Left_thumb_Pos, 'front')
            # right_wrist_twist_angle = calculate_angle(Right_index_Pos, Right_wrist_Pos, Right_thumb_Pos, 'front')

            step4_left_score = 2
            step4_right_score = 2


            # Table A score


            # Step 9 - side view neck position
            neck_angle = findAngle(Neck_Pos[0], Neck_Pos[1], Nose_Pos[0], Nose_Pos[1]) - 30
            # neck_angle = calculate_angle((Neck_Pos[0], Neck_Pos[1] - 0.1, Neck_Pos[2]), Neck_Pos, Nose_Pos, 'front') + 60
            print("neck_kuy = " + str(neck_angle))
            if 0 <= neck_angle < 10.5:
                neck_score = 1
            elif 10.5 <= neck_angle < 20.5:
                neck_score = 2         
            elif 20.5 <= neck_angle:
                neck_score = 3
            else:
                neck_score = 4
            print("neck score = " + str(neck_score))
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


            #step 10
            #print ("Cali AN = "+str(calibrate_angle))
            #print("view2 = " + str(view2))
            if view2 == -1:
                trunk_angle =180-calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'front')
    
            if view2 == 1:
                trunk_angle=180 - calculate_angle(Left_knee_Pos, Left_hip_Pos, Left_shoulder_Pos, 'front')
            

            if view2 != 0 :
                if 0 < trunk_angle <= 8.3:
                        trunk_score = 1
                elif 8.3 < trunk_angle <= 20:
                        trunk_score = 2
                elif 20 <= trunk_angle <= 60:
                        trunk_score = 3
                elif 60 < trunk_angle:
                        trunk_score = 4
                
            if view2 == 0:
                trunk_angle=0
                trunk_score=1
            # Addition bending
            # trunk_bent_angle = calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'front')
            # if 180 >= trunk_bent_angle > 162.5 :
            #     trunk_bent_score = 0
            # else:
            #     trunk_bent_score = 1
            # step10_score = trunk_score + trunk_bent_score
            step10_score = trunk_score
            #print("step10 trunk angle = " + str(trunk_angle))
            #print("step10 trunk score = " + str(step10_score))

            # Step 11 - side&front views legs position
            left_knee_angle = calculate_angle(Left_hip_Pos, Left_knee_Pos, Left_ankle_Pos, 'side')
            right_knee_angle = calculate_angle(Right_hip_Pos, Right_knee_Pos, Right_ankle_Pos, 'side')
            if abs(left_knee_angle - right_knee_angle) >= 10 or abs(Left_ankle_Pos[1] - Right_ankle_Pos[1]) > abs(Hip_Pos[1] - Neck_Pos[1])/2.4: # Unbalance
                legs_score = 2
            else: # Balance
                legs_score = 1

            step11_score = legs_score
            #print("step11 leg score = " + str(step11_score))

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
            
            
            #cv2.imshow("Proccesing", image_with_keypoints)
            
            
            

        else:
            image_with_keypoints = frame.copy() 
           #cv2.imshow("Image with Keypoints", frame)

    
        ###############test
        frames.append(image_with_keypoints)
        ###########################
        # Check for 'q' key press to exit
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        print("Left RULA grand score = " + str(LC))
        print("Right RULA grand score = " + str(RC))
        variable1.set("LC : " + str(LC))
        variable2.set("RC : " + str(RC))
        root.update()
        
        
            
    

    # Release the webcam and close all OpenCV windows
    cap.release()
    #Test
    
    # for frame in frames:
    #     output_video.write(image_with_keypoints)  

    
    
    
    cv2.destroyAllWindows()
    #Test
   
    # Reconstruct the video
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]
    fps = 30  # Adjust the frame rate as needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
    for image_with_keypoints in frames:
        #cv2.imshow('Reconstructed Video', image_with_keypoints)
        cv2.waitKey(30)
        out.write(image_with_keypoints)  # Adjust delay as needed for desired frame r
        

    print("Video reconstruction complete.")
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Close MediaPipe pose model
    pose.close()

def image_pose_estimation(name):
    # Initialize MediaPipe pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

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
        (26, 28),   # Right knee to right ankle

        (11, 12),  # Left shoulder to right shoulder

        (8, 6),  
        (6, 5),   
        (5, 4),  
        (4, 0),    
        (0, 1),    
        (1, 2),    
        (2, 3),  
        (3, 7),
        (10, 9),
        (15,19),
        (16,20)
    ]

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
    

    global handswrist
    global Index_Tip
    global Thumb_CMC
    global Thumb_MCP
    global Thumb_IP
    global Thumb_Tip
    global Index_MCP
    global Index_PIP
    global Index_DIP
    global Middle_MCP
    global Middle_PIP
    global Middle_DIP
    global Middle_Tip
    global Ring_MCP
    global Ring_PIP
    global Ring_DIP
    global Ring_Tip
    global Pinky_MCP
    global Pinky_PIP
    global Pinky_DIP
    global Pinky_Tip


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
    

    while True:
        
        ########reset######
        Pinky_Tip=None
        Thumb_Tip=None 
        handswrist=None
        #################33
        ret, frame = cap.read()
        if not ret:
            break
   
        # Detect keypoints
        pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #forhand
        hands_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # If pose detected, plot keypoints and lines
        if pose_results.pose_landmarks :
            image_with_keypoints = frame.copy()  # Create a copy of the captured image
            keypoints_3d = []
            mp_drawing.draw_landmarks(image_with_keypoints,pose_results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

            for landmark in pose_results.pose_landmarks.landmark:
                cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z
                
                # Store keypoints
                keypoints_3d.append((cx, cy, cz))

            Nose_Pos = keypoints_3d[0]
            Left_eye_inner_Pos = keypoints_3d[1]
            Left_eye_Pos = keypoints_3d[2]
            Left_eye_outer_Pos = keypoints_3d[3]
            Right_eye_inner_Pos = keypoints_3d[4]
            Right_eye_Pos = keypoints_3d[5]
            Right_eye_outer_Pos = keypoints_3d[6]
            Left_ear_Pos = keypoints_3d[7]
            Right_ear_Pos = keypoints_3d[8]
            Mouth_left_Pos = keypoints_3d[9]
            Mouth_right_Pos = keypoints_3d[10]
            Left_shoulder_Pos = keypoints_3d[11]
            Right_shoulder_Pos = keypoints_3d[12]
            Left_elbow_Pos = keypoints_3d[13]
            Right_elbow_Pos = keypoints_3d[14]
            Left_wrist_Pos = keypoints_3d[15]
            Right_wrist_Pos = keypoints_3d[16]
            Left_pinky_Pos = keypoints_3d[17]
            Right_pinky_Pos = keypoints_3d[18]
            Left_index_Pos = keypoints_3d[19]
            Right_index_Pos = keypoints_3d[20]
            Left_thumb_Pos = keypoints_3d[21]
            Right_thumb_Pos = keypoints_3d[22]
            Left_hip_Pos = keypoints_3d[23]
            Right_hip_Pos = keypoints_3d[24]
            Left_knee_Pos = keypoints_3d[25]
            Right_knee_Pos = keypoints_3d[26]
            Left_ankle_Pos = keypoints_3d[27]
            Right_ankle_Pos = keypoints_3d[28]
            Left_heel_Pos = keypoints_3d[29]
            Right_heel_Pos = keypoints_3d[30]
            Left_foot_index_Pos = keypoints_3d[31]
            Right_foot_index_Pos = keypoints_3d[32]




            # Calculate additional positions
            Neck_Pos = ((Right_shoulder_Pos[0] + Left_shoulder_Pos[0]) / 2,
                        (Right_shoulder_Pos[1] + Left_shoulder_Pos[1]) / 2,
                        (Right_shoulder_Pos[2] + Left_shoulder_Pos[2]) / 2)

            Head_Pos = ((Right_ear_Pos[0] + Left_ear_Pos[0]) / 2,
                        (Right_ear_Pos[1] + Left_ear_Pos[1]) / 2,
                        (Right_ear_Pos[2] + Left_ear_Pos[2]) / 2)

            Hip_Pos = ((Right_hip_Pos[0] + Left_hip_Pos[0]) / 2,
                    (Right_hip_Pos[1] + Left_hip_Pos[1]) / 2,
                    (Right_hip_Pos[2] + Left_hip_Pos[2]) / 2)

            # Separate 3D coordinates into individual lists
            keypoints_x, keypoints_y, keypoints_z = zip(*keypoints_3d)

            # Draw lines between keypoints
            for connection in keypoint_connections:
                start_point = connection[0]
                end_point = connection[1]
                start_coords = (int(keypoints_3d[start_point][0]), int(keypoints_3d[start_point][1]))
                end_coords = (int(keypoints_3d[end_point][0]), int(keypoints_3d[end_point][1]))
                cv2.line(image_with_keypoints, start_coords, end_coords, color=(255, 0, 0), thickness=2)  # Red line

                #forhand
            if hands_results.multi_hand_landmarks:
                handskeypoints_3d = []
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image_with_keypoints,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                     mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()) 
                    for landmark in hand_landmarks.landmark:
                        cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z
                        handskeypoints_3d.append((cx, cy, cz))
                    
                    handswrist = handskeypoints_3d[0]
                    Thumb_CMC = handskeypoints_3d[1]
                    Thumb_MCP = handskeypoints_3d[2]
                    Thumb_IP = handskeypoints_3d[3]
                    Thumb_Tip = handskeypoints_3d[4]
                    Index_MCP = handskeypoints_3d[5]
                    Index_PIP = handskeypoints_3d[6]
                    Index_DIP = handskeypoints_3d[7]
                    Index_Tip = handskeypoints_3d[8]
                    Middle_MCP = handskeypoints_3d[9]
                    Middle_PIP = handskeypoints_3d[10]
                    Middle_DIP = handskeypoints_3d[11]
                    Middle_Tip = handskeypoints_3d[12]
                    Ring_MCP = handskeypoints_3d[13]
                    Ring_PIP = handskeypoints_3d[14]
                    Ring_DIP = handskeypoints_3d[15]
                    Ring_Tip = handskeypoints_3d[16]
                    Pinky_MCP = handskeypoints_3d[17]
                    Pinky_PIP = handskeypoints_3d[18]
                    Pinky_DIP = handskeypoints_3d[19]
                    Pinky_Tip = handskeypoints_3d[20]


            # if  Index_Tip != None :
            #     Index_text = "Index Tip"
            #     Index_text_pos = (int(Index_Tip[0]), int(Index_Tip[1]))
            #     cv2.putText(image_with_keypoints, Index_text, Index_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # if  Pinky_Tip != None :
                # Pinky_text = "Pinky Tip"
                # Pinky_text_pos = (int(Pinky_Tip[0]), int(Pinky_Tip[1]))
                # cv2.putText(image_with_keypoints, Pinky_text, Pinky_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print("hand pinky  index   = " + str(Pinky_Tip))     
            text_posx = 20
            text_step = 40
            

            
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

            if 45 <= calibrate_angle < 120:
                view2 = 0 # Up
            elif 0 <= calibrate_angle < 45:
                view2 = 1 # Left
            elif 120 <= calibrate_angle < 180:
                view2 = -1 # Right
            
            # Calculate angle
                    
            #Tellside step test
            bodysideangle=calculate_angle(Left_shoulder_Pos, Right_shoulder_Pos, (Right_shoulder_Pos[0], Right_shoulder_Pos[1],Right_shoulder_Pos[2]-1), 'top')
            if 65 <=bodysideangle<= 110:
                #หันเข้ากล้อง
                Tellside = 0
            else :
                #หันข้าง
                Tellside = 1

            cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(bodysideangle)), (10, text_posx*5), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(Tellside)), (10, text_posx*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
        
                    
            # Step 1 - side view shoulder position
            #หันหน้า
            if Tellside == 0: 
              
                #step1
                left_shoulder_angle= calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'side')
                right_shoulder_angle= calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'side')
                left_shoulder_abduct_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'front')
                right_shoulder_abduct_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'front')
                
                #step2
                left_elbow_angle = calculate_angle(Left_shoulder_Pos, Left_elbow_Pos, Left_wrist_Pos, 'side')
                right_elbow_angle = calculate_angle(Right_wrist_Pos, Right_elbow_Pos, Right_shoulder_Pos, 'side')

                

                if  right_shoulder_abduct_angle > 90 :
                    right_shoulder_angle=10

                if left_shoulder_abduct_angle> 90 :
                    left_shoulder_angle=10

            #พึ่งเพิ่ม step3 กัน error
                if Pinky_Tip == None :
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'side'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'side'))
                if Pinky_Tip != None:
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Pinky_Tip, 'side'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos,Pinky_Tip, 'side'))

            #หันข้าง
            #if view2 == 1 :
            if Tellside == 1:
                #step1
                left_shoulder_angle= calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'front')
                right_shoulder_angle= calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'front')
                left_shoulder_abduct_angle = calculate_angle(Left_elbow_Pos, Left_shoulder_Pos, Left_hip_Pos, 'side')
                right_shoulder_abduct_angle = calculate_angle(Right_elbow_Pos, Right_shoulder_Pos, Right_hip_Pos, 'side')
                #step2
                left_elbow_angle =180 - calculate_angle(Left_shoulder_Pos, Left_elbow_Pos, Left_wrist_Pos, 'front')
                right_elbow_angle = 180 - calculate_angle(Right_shoulder_Pos, Right_elbow_Pos, Right_wrist_Pos, 'front')
                #Righi left shoulder abduct can be 0
                if  right_shoulder_angle > 90 :
                    right_shoulder_abduct_angle=10

                if left_shoulder_angle> 90 :
                    left_shoulder_abduct_angle=10

            #step3 (Accu
              
            #   left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'front'))
            #   right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'front'))
              
                                #step3 (Accu75)
                #จับมือไม่ได้
                if Pinky_Tip == None :
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Left_index_Pos, 'front'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'front'))
                    
                    #step4วันนี้
                    step4_angleright=None
                    step4_angleleft=None
                    step4_left_score = 2
                    step4_right_score =2

                #จับมือได้
                if Pinky_Tip != None:
                    left_wrist_angle = abs(180-calculate_angle(Left_elbow_Pos, Left_wrist_Pos,Pinky_Tip, 'front'))
                    right_wrist_angle = abs(180-calculate_angle(Right_elbow_Pos, Right_wrist_Pos,Pinky_Tip, 'front'))
                    
                    #step4วันนี้
                    step4_angleleft = calculate_angle(Left_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                    if step4_angleleft < 90 :
                     step4_left_score=  1
                    else :
                     step4_left_score= 2
                    
                    step4_angleright = calculate_angle(Right_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                    
                    if step4_angleright < 90 :
                     step4_right_score=  1
                    else :
                     step4_right_score= 2
                    
                    

                print("step4ang right = "+str(step4_angleright))
                print("step4ang left = "+str(step4_angleleft))
                print("step4scoretestright =  "+ str(step4_right_score))
                print("step4scoretestleft =  "+ str(step4_left_score))
                cv2.putText(image_with_keypoints, "step4testleft: " + str("{:0.2f}".format(step4_left_score)), (10, text_posx+text_step*9), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                cv2.putText(image_with_keypoints, "step4testright : " + str("{:0.2f}".format(step4_right_score)), (10, text_posx+text_step*10), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                    

                #Reset value
                
                # #step 4 test
                # print(Thumb_Tip)
                # print(Pinky_Tip)
                # step4_angleright = calculate_angle(Right_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                # if step4_angleright < 90 :
                #      Teststep4_right_score =  1
                # else :
                #      Teststep4_right_score= 2
                # step4_angleleft = calculate_angle(Left_wrist_Pos,Thumb_Tip,Pinky_Tip,'front')
                # if step4_angleleft < 90 :
                #      Teststep4_left_score =  1
                # else :
                #      Teststep4_left_score= 2

                  
                #   print("view2 = " + str(view2))
                #   print("hand  index   = " + str(Index_Tip))
              
                print("pose  index  = " + str(Right_index_Pos))
                cv2.putText(image_with_keypoints, "L_wrist_angle : " + str("{:0.2f}".format(left_wrist_angle)), (10, text_posx+text_step*7), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
                cv2.putText(image_with_keypoints, "R_wrist_angle : " + str("{:0.2f}".format(right_wrist_angle)), (10, text_posx+text_step*8), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)


            
              

            #   if Left_index_Pos is None:
            #         print("Left hand index not detected")
            #         # If it's None, display "Cant detect"
            #         cv2.putText(image_with_keypoints, "Left hand can't detect" , (10, text_posx+text_step*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #   else:
            #         print("Left hand index detected")
            #         # If it's not None, display "Can detect"
            #         cv2.putText(image_with_keypoints, "can detect" , (10, text_posx+text_step*6), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)


                                    
            #Step1
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
            #print("step1 left score = " + str(step1_left_score) + " and step1 right score = " + str(step1_right_score))
            
            
            
            
            
            
            #cv2.putText(image_with_keypoints, "Left base side : " + str("{:0.2f}".format(Tellside)), (10, text_posx), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            
            
            cv2.putText(image_with_keypoints, "L_upper_angle : " + str("{:0.2f}".format(left_shoulder_angle)), (10, text_posx), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            #cv2.putText(image_with_keypoints, "L_upper_arm_score : " + str("{:0.2f}".format(left_shoulder_score)), (300, text_posx),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            cv2.putText(image_with_keypoints, "R_upper_angle : " + str("{:0.2f}".format(right_shoulder_angle)), (10, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            #cv2.putText(image_with_keypoints, "R_upper_score : " + str("{:0.2f}".format(right_shoulder_score)), (300, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            
            cv2.putText(image_with_keypoints, "L_upper_abduct angle : " + str("{:0.2f}".format(left_shoulder_abduct_angle)), (300, text_posx),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            cv2.putText(image_with_keypoints, "R_upper_abduct angle : " + str("{:0.2f}".format(right_shoulder_abduct_angle)), (300, text_posx+text_step),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)

            cv2.putText(image_with_keypoints, "L_wrist_score : " + str("{:0.2f}".format(left_shoulder_score)), (800, text_posx), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "R_wrist_score : " + str("{:0.2f}".format(right_shoulder_score)), (800, text_posx+text_step), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2) 

            print("step1 left score = " + str(step1_left_score) + " and step1 right score = " + str(step1_right_score))


            # Step 2 - side view elbow position
          
            if 60<= left_elbow_angle <= 105:
                left_elbow_score = 1
            else:
                left_elbow_score = 2
            if 60 <= right_elbow_angle < 105:
                right_elbow_score = 1
            else:
                right_elbow_score = 2           
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

            step2_left_score = left_elbow_score + wrist_midline
            step2_right_score = right_elbow_score + wrist_midline
            print("step2 left score = " + str(step2_left_score) + " and step2 right score = " + str(step2_right_score))
            cv2.putText(image_with_keypoints, "L_lower_angle : " + str("{:0.2f}".format(left_elbow_angle)), (10, text_posx+text_step*3), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "R_lower_angle : " + str("{:0.2f}".format(right_elbow_angle)), (10, text_posx+text_step*4), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "L_Lower_score : " + str("{:0.2f}".format(left_elbow_score)), (600, text_posx+text_step*3), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "R_Lower_score : " + str("{:0.2f}".format(right_elbow_score)), (600, text_posx+text_step*4), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2) 


            # Step 3 - side view wrist position
            # left_wrist_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'side')
            # right_wrist_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'side')
            if 0 <=left_wrist_angle <=5:
                left_wrist_score=1
            elif 5<left_wrist_angle<=15:
                left_wrist_score=2
            elif 15<=left_wrist_angle:
                left_wrist_score=3
            
            if 0 <=right_wrist_angle <=5:
                right_wrist_score=1
            elif 5<right_wrist_angle<=15:
                right_wrist_score=2
            elif 15<=right_wrist_angle:
                right_wrist_score=3
            
             
            


            # Addition - top view wrist deviation
            left_wrist_deviation_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'top')
            right_wrist_deviation_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'top')

            step3_left_score = left_wrist_score
            step3_right_score = right_wrist_score
            # step3_left_score = 2
            # step3_right_score = 2
            # print("hand right index   = " + str(Index_Tip))
            # print("pose right index  = " + str(Right_index_Pos))
            print("lefttwrist score= " + str(step3_left_score))
            print("rightwrist score = " + str(step3_right_score))
            cv2.putText(image_with_keypoints, "L_wrist_score : " + str("{:0.2f}".format(left_wrist_score)), (600, text_posx+text_step*7), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            cv2.putText(image_with_keypoints, "R_wrist_score : " + str("{:0.2f}".format(right_wrist_score)), (600, text_posx+text_step*8), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2) 
            # Step 4 - front view wrist twist
            left_wrist_twist_angle = calculate_angle(Left_index_Pos, Left_wrist_Pos, Left_thumb_Pos, 'front')
            right_wrist_twist_angle = calculate_angle(Right_index_Pos, Right_wrist_Pos, Right_thumb_Pos, 'front')

            step4_left_score = 2
            step4_right_score = 2


            # Table A score


            # Step 9 - side view neck position
            neck_angle = findAngle(Neck_Pos[0], Neck_Pos[1], Nose_Pos[0], Nose_Pos[1]) - 30
            # neck_angle = calculate_angle((Neck_Pos[0], Neck_Pos[1] - 0.1, Neck_Pos[2]), Neck_Pos, Nose_Pos, 'front') + 60
            print("neck_ = " + str(neck_angle))
            if 0 <= neck_angle < 10.5:
                neck_score = 1
            elif 10.5 <= neck_angle < 20.5:
                neck_score = 2         
            elif 20.5 <= neck_angle:
                neck_score = 3
            else:
                neck_score = 4
            print("neck score = " + str(neck_score))
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
            # print("step9 neck angle = " + str(neck_angle))
            # print("step9 neck score = " + str(step9_score))


            # Step 10 - side view trunk position
            #trunk_angle = calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'side')
            #step 10
            print ("Cali AN = "+str(calibrate_angle))
            print("view2 = " + str(view2))
            #Right
            if view2 == -1:  
                #neck_angle = calculate_angle(Right_hip_Pos, Right_shoulder_Pos, Nose_Pos, 'front')
                trunk_angle =180-calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'front')
    
            #Left
            if view2 == 1:
                #neck_angle = calculate_angle(Left_hip_Pos, Left_shoulder_Pos, Nose_Pos, 'front')
                trunk_angle=180 - calculate_angle(Left_knee_Pos, Left_hip_Pos, Left_shoulder_Pos, 'front')
            

            if view2 != 0 :
                if 0 < trunk_angle <= 8.3:
                        trunk_score = 1
                elif 8.3 < trunk_angle <= 20:
                        trunk_score = 2
                elif 20 <= trunk_angle <= 60:
                        trunk_score = 3
                elif 60 < trunk_angle:
                        trunk_score = 4
                
            if view2 == 0:
                trunk_angle=0
                trunk_score=1
            # Addition bending
            # trunk_bent_angle = calculate_angle(Right_knee_Pos, Right_hip_Pos, Right_shoulder_Pos, 'front')
            # if 180 >= trunk_bent_angle > 162.5 :
            #     trunk_bent_score = 0
            # else:
            #     trunk_bent_score = 1
            # step10_score = trunk_score + trunk_bent_score
            step10_score = trunk_score
            print("step10 trunk angle = " + str(trunk_angle))
            print("step10 trunk score = " + str(step10_score))

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

            cv2.imshow("Image with Keypoints", image_with_keypoints)  
            # Check for 'q' key press to exit
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
            print("Left RULA grand score = " + str(LC))
            print("Right RULA grand score = " + str(RC))
            variable1.set("LC : " + str(LC))
            variable2.set("RC : " + str(RC))
            root.update()

    # Close MediaPipe pose model
    pose.close()

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

# def conditionMuscle2():
#     global Muscle
#     options1 = ['No', 'Yes']
#     Muscle = st.selectbox("Is the posture mainly static or action repeated occurs?", options1)
#     # st.write("Muscle = " + str(Muscle))
#     return Muscle

# def conditionWeight2():
#     global Weight
#     values = list(range(0, 16))
#     Weight = st.select_slider("What is the weight of the load?", options = values)
#     # st.write("Weight = " + str(Weight))
#     return Weight



def webcam():
    webcam2(0)

def browse():
        filename = filedialog.askopenfilename()
        mimestart = mimetypes.guess_type(str(filename))[0]

        if mimestart != None:
            mimestart = mimestart.split('/')[0]
        if mimestart == 'video':
            video_pose_estimation(str(filename))
        elif mimestart == 'image':
            image_pose_estimation(str(filename))
        else:
            pass

l1 =Label(root, text = "Biomechanic Posture System", font= ('Helvetica 25 bold', 30), fg = "white", bg = "#242424").place(relx=.5, rely=0.05,anchor= N)
l2 =Label(root, textvariable = variable1, font= ('Helvetica 10 bold', 25), fg = "white", bg = "#242424").place(relx=.5, rely=.775,anchor= N)
l3 =Label(root, textvariable = variable2, font= ('Helvetica 10 bold', 25), fg = "white", bg = "#242424").place(relx=.5, rely=.85,anchor= N)

b1 = ctk.CTkButton(master = root, text = "Webcam", command = webcam).place(relx=0.5, rely=.1625, anchor = CENTER) 
b2 = ctk.CTkButton(master = root, text = "Browse", command = browse).place(relx=0.5, rely=.25, anchor = CENTER)

root. mainloop()