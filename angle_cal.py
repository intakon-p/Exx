import numpy as np

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


P1 = (255.28249740600586, 341.40904426574707, -0.09098196029663086)
P2 = (353.79103660583496, 351.4533519744873, -0.08191635087132454)
P3 = (317.008056640625, 247.252836227417, -0.53230220079422)

calculate_angle_between_keypoints_2d(P1, P2, P3)
print("Angle = " + str(angle))

# Example usage:
# Assuming keypoints are tuples containing (x, y, z) coordinates
# keypoint1 = (x1, y1, z1)
# keypoint2 = (x2, y2, z2)
# keypoint3 = (x3, y3, z3)
# angle = calculate_angle_between_keypoints_2d(keypoint1, keypoint2, keypoint3)
# print("Angle between keypoint1, keypoint2, and keypoint3 projected to 2D:", angle)
