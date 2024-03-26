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

# Create a new 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set initial axis limits
ax.set_xlim([0, 640])  # Adjust according to your image size
ax.set_ylim([480, 0])  # Adjust according to your image size
ax.set_zlim([0, 1])    # Fixed z-limit for 3D visualization

def DimGraphCam(capName):
    # Open the webcam
    cap = cv2.VideoCapture(capName)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose estimation
        results = pose.process(rgb_frame)
        
        # Clear the plot for the next frame
        ax.clear()
        
        # If pose detected, plot keypoints and lines
        if results.pose_landmarks:
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



            # # Merge both lists of connections
            # all_connections = keypoint_connections + keypoint_connections_with_midpoint

            # # Draw lines between connected keypoints with specified color
            # for connection in all_connections:
            #     start_point_idx, end_point_idx = connection
            #     start_point = keypoints_3d[start_point_idx]
            #     end_point = keypoints_3d[end_point_idx]
            #     if connection in connection_colors:
            #         color = connection_colors[connection]
            #     else:
            #         color = 'gray'  # Default to gray if color not specified
            #     ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=color)


            # Update plot
            plt.pause(.5)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Close MediaPipe pose model
    pose.close()


def DimGraphImg(ImgName):
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
        
        # Clear the plot for the next frame
        ax.clear()
        
        # If pose detected, plot keypoints and lines
        if results.pose_landmarks:
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



            # # Merge both lists of connections
            # all_connections = keypoint_connections + keypoint_connections_with_midpoint

            # # Draw lines between connected keypoints with specified color
            # for connection in all_connections:
            #     start_point_idx, end_point_idx = connection
            #     start_point = keypoints_3d[start_point_idx]
            #     end_point = keypoints_3d[end_point_idx]
            #     if connection in connection_colors:
            #         color = connection_colors[connection]
            #     else:
            #         color = 'gray'  # Default to gray if color not specified
            #     ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=color)


            # Update plot
            plt.pause(.5)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Close MediaPipe pose model
    pose.close()



DimGraphCam(0)