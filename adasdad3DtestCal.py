import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

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



def calculate_angle_between_keypoints_2d(keypoint1, keypoint2, keypoint3):
    global angle
    # Only use x and y coordinates for angle calculation
    x1, y1, _ = keypoint1
    x2, y2, _ = keypoint2
    x3, y3, _ = keypoint3
    
    # Calculate vectors between keypoints
    vector1 = np.array([x1 - x2, y1 - y2])
    vector2 = np.array([x3 - x2, y3 - y2])

    # Calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    # Calculate cosine of the angle
    cos_theta = dot_product / (magnitude_vector1 * magnitude_vector2)

    # Convert cosine to angle in degrees
    angle = np.degrees(np.arccos(cos_theta))
    
    return angle

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

# Create a new figure for subplots
fig = plt.figure(figsize=(15, 5))

# Subplot 2
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title('Top')
ax2.view_init(elev=0, azim=-90, roll=180)

# Subplot 1
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title('Front')
ax1.view_init(elev=-90, azim=-90, roll=0)

# Subplot 3
ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title('Side')
ax3.view_init(elev=0, azim=-180, roll=90)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform pose estimation
    results = pose.process(rgb_frame)
    
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
            
            # Add the midpoint shoulders to keypoints_3d
            keypoints_3d.append(midpoint_shoulders)

            # Add the midpoint shoulders to keypoint_colors
            keypoint_colors[len(keypoints_3d) - 1] = '#ce4802'  # Green color for midpoint shoulders

            # Define connections between the midpoint and other keypoints
            keypoint_connections_with_midpoint = [
                # Midpoint shoulders to nose
                (len(keypoints_3d) - 1, Pose.NOSE.value),
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

              # Plot lines connecting the midpoint to other keypoints
            for connection in keypoint_connections_with_midpoint:
                start_point_idx, end_point_idx = connection
                start_point = keypoints_3d[start_point_idx]
                end_point = keypoints_3d[end_point_idx]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color='#0d0096')

            # Define connections between the midpoint and each shoulder with colors
            keypoint_connections_with_midpoint = [
                # Midpoint shoulders to left shoulder (color: blue)
                (len(keypoints_3d) - 1, Pose.LEFT_SHOULDER.value, '#d44606'),
                # Midpoint shoulders to right shoulder (color: red)
                (len(keypoints_3d) - 1, Pose.RIGHT_SHOULDER.value, '#970102'),
            ]

            # Plot lines connecting the midpoint to each shoulder with different colors
            for connection in keypoint_connections_with_midpoint:
                start_point_idx, end_point_idx, color = connection
                start_point = keypoints_3d[start_point_idx]
                end_point = keypoints_3d[end_point_idx]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=color)

            # Print all keypoint positions
            print("Keypoint positions:")
            for idx, point_3d in enumerate(keypoints_3d):
                print(f"Keypoint {idx}: {point_3d}")
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

            # Print the positions of specific keypoints
            print("Nose position:", Nose_Pos)
            print("Right Shoulder position:", Right_shoulder_Pos)
            print("Left Shoulder position:", Left_shoulder_Pos)
            print("Left Shoulder position:", Left_foot_index_Pos)

    plt.pause(30)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close MediaPipe pose model
pose.close()

plt.show()
