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

# Create a new 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set initial axis limits
ax.set_xlim([0, 640])  # Adjust according to your image size
ax.set_ylim([480, 0])  # Adjust according to your image size
ax.set_zlim([0, 1])    # Fixed z-limit for 3D visualization

# Open the webcam
cap = cv2.VideoCapture(0)

# T - elev = 0, azim = 90, roll = 0
# F - elev = 90, azim = 90, roll = 0
# F - elev = 90, azim = 90, roll = 90

# Initial view angles
elev = 90
azim = 90
roll = 90

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect keypoints
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Clear the plot for the next frame
    ax.clear()
    
    # If pose detected, plot keypoints and lines
    if results.pose_landmarks:
        keypoints_x = []
        keypoints_y = []
        keypoints_z = []

        for landmark in results.pose_landmarks.landmark:
            cx, cy, cz = landmark.x * frame.shape[1], landmark.y * frame.shape[0], 0.0
            
            # Store keypoints
            keypoints_x.append(cx)
            keypoints_y.append(cy)
            keypoints_z.append(cz)
        
        # Plot keypoints
        ax.scatter(keypoints_x, keypoints_y, keypoints_z, c='r', marker='o')  

        # Draw lines between keypoints
        for connection in keypoint_connections:
            start_point = connection[0]
            end_point = connection[1]
            ax.plot([keypoints_x[start_point], keypoints_x[end_point]],
                    [keypoints_y[start_point], keypoints_y[end_point]],
                    [keypoints_z[start_point], keypoints_z[end_point]], 'b-')

        # Update plot
        plt.pause(0.01)

    # Update view angles (rotate around z-axis)
    # elev += 1  # Increment azimuth angle to rotate around z-axis
    # ax.view_init(elev=elev, azim=azim, roll=roll)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close MediaPipe pose model
pose.close()
