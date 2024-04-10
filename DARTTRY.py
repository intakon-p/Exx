import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a new figure for subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Function to update the plot
def update_plot():
    # Clear axes
    ax1.clear()
    ax2.clear()

    # If pose detected, plot keypoints and lines
    if pose_results.pose_landmarks:
        # Extract pose landmarks
        pose_landmarks = []
        for landmark in pose_results.pose_landmarks.landmark:
            x = landmark.x * frame.shape[1]
            y = landmark.y * frame.shape[0]
            z = landmark.z * frame.shape[1]  # Depth (distance from the camera)
            pose_landmarks.append((x, y, z))

        # Plot pose landmarks
        for landmark in pose_landmarks:
            ax1.scatter(landmark[0], landmark[1], landmark[2], c='r')

    # If hand detected, plot hand landmarks
    if hand_results.multi_hand_landmarks:
        # Extract hand landmarks
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                z = landmark.z * frame.shape[1]  # Depth (distance from the camera)
                ax1.scatter(x, y, z, c='b')

    # Show the original frame with detected landmarks
    ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax2.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.pause(0.001)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose landmarks
    pose_results = pose.process(frame_rgb)

    # Detect hand landmarks
    hand_results = hands.process(frame_rgb)

    # Update the plot
    update_plot()

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
