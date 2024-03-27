import numpy as np

def calculate_angle(keypoint1, keypoint2, keypoint3, plane):
    global angle
    # Extract x, y, and z coordinates for each keypoint
    x1, y1, z1 = keypoint1
    x2, y2, z2 = keypoint2
    x3, y3, z3 = keypoint3
    
    if plane == 'xy':
        # Calculate vectors between keypoints in the XY-plane
        vector1 = np.array([x1 - x2, y1 - y2, 0])
        vector2 = np.array([x3 - x2, y3 - y2, 0])
    elif plane == 'xz':
        # Calculate vectors between keypoints in the XZ-plane
        vector1 = np.array([x1 - x2, 0, z1 - z2])
        vector2 = np.array([x3 - x2, 0, z3 - z2])
    elif plane == 'zx':
        # Calculate vectors between keypoints in the ZX-plane
        vector1 = np.array([x1 - x2, 0, z1 - z2])
        vector2 = np.array([x3 - x2, 0, z3 - z2])

    # Calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    # Calculate cosine of the angle
    cos_theta = dot_product / (magnitude_vector1 * magnitude_vector2)

    # Convert cosine to angle in degrees
    angle = np.degrees(np.arccos(cos_theta))
    
    return angle