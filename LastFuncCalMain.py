import cv2
import mediapipe as mp
from angle_cal import *
import pandas as pd


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
    pose = mp_pose.Pose()

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
   
        # Detect keypoints
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # If pose detected, plot keypoints and lines
        if results.pose_landmarks:
            image_with_keypoints = frame.copy()  # Create a copy of the captured image
            keypoints_3d = []

            for landmark in results.pose_landmarks.landmark:
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
            # print("step1 left score = " + str(step1_left_score) + " and step1 right score = " + str(step1_right_score))


            # Step 2 - side view elbow position
            left_wrist_angle = calculate_angle(Left_shoulder_Pos, Left_elbow_Pos, Left_wrist_Pos, 'side')
            right_wrist_angle = calculate_angle(Right_shoulder_Pos, Right_elbow_Pos, Right_wrist_Pos, 'side')   
            if 90 <= left_wrist_angle <= 150:
                left_wrist_score = 1
            else:
                left_wrist_score = 2
            if 90 <= right_wrist_angle < 150:
                right_wrist_score = 1
            else:
                right_wrist_score = 2           
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

            step2_left_score = left_wrist_score + wrist_midline
            step2_right_score = right_wrist_score + wrist_midline
            # print("step2 left score = " + str(step2_left_score) + " and step2 right score = " + str(step2_right_score))


            # Step 3 - side view wrist position
            left_wrist_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'side')
            right_wrist_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'side')
            # Addition - top view wrist deviation
            left_wrist_deviation_angle = calculate_angle(Left_elbow_Pos, Left_wrist_Pos, Left_index_Pos, 'top')
            right_wrist_deviation_angle = calculate_angle(Right_elbow_Pos, Right_wrist_Pos, Right_index_Pos, 'top')

            step3_left_score = 2
            step3_right_score = 2


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


            # Step 11 - side&front views legs position
            left_knee_angle = calculate_angle(Left_hip_Pos, Left_knee_Pos, Left_ankle_Pos, 'side')
            right_knee_angle = calculate_angle(Right_hip_Pos, Right_knee_Pos, Right_ankle_Pos, 'side')
            if abs(left_knee_angle - right_knee_angle) >= 10 or abs(Left_ankle_Pos[1] - Right_ankle_Pos[1]) > abs(Hip_Pos[1] - Neck_Pos[1])/2.4: # Unbalance
                legs_score = 2
            else: # Balance
                legs_score = 1

            step11_score = legs_score

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
            print("Left RULA grand score = " + str(LC))
            print("Right RULA grand score = " + str(RC))

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

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