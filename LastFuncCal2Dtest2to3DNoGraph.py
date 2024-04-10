import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define indices of keypoints for drawing lines
keypoint_connections = [
    (11, 12),  # Left shoulder to right shoulder
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
cap = cv2.VideoCapture(0)

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
            
        cv2.imshow("Image with Keypoints", image_with_keypoints)   
        print("Nose1 = " + str(keypoints_3d[0]))
        print("Nose2 = " + str(Nose_Pos))
        print("Hip = " + str(Hip_Pos))

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close MediaPipe pose model
pose.close()

plt.show()
