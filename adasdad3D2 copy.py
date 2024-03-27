import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Subplot 1
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title('Front')
ax1.view_init(elev=-90, azim=-90, roll=0)
ax1.axis('off')

# Subplot 2
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title('Top')
ax2.view_init(elev=0, azim=-90, roll=180)
ax2.axis('off')

# Subplot 3
ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title('Side')
ax3.view_init(elev=0, azim=-180, roll=90)
ax3.axis('off')

# Plot the keypoints and connections
for ax in [ax1, ax2, ax3]:
    # Plot keypoints in 3D space with specified color
    for idx, color in keypoint_colors.items():
        ax.scatter(0, 0, 0, c=color)  # Dummy point for legend
        
    # Draw lines between connected keypoints with specified color
    for connection, color in connection_colors.items():
        start_point_idx, end_point_idx = connection
        ax.plot([0, 0], [0, 0], [0, 0], color=color)  # Dummy line for legend

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
            # Plot keypoints in 3D space with specified color
            for landmark in results.pose_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z
                ax.scatter(cx, cy, cz, c='red', marker='o')

            # Draw lines between connected keypoints with specified color
            for connection in keypoint_connections:
                start_point_idx, end_point_idx = connection
                start_point = results.pose_landmarks.landmark[start_point_idx]
                end_point = results.pose_landmarks.landmark[end_point_idx]
                start_x, start_y, start_z = start_point.x * frame.shape[1], start_point.y * frame.shape[0], start_point.z
                end_x, end_y, end_z = end_point.x * frame.shape[1], end_point.y * frame.shape[0], end_point.z
                ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], c='blue')

    plt.pause(0.01)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close MediaPipe pose model
pose.close()

plt.show()

