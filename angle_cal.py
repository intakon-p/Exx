import numpy as np
import math as m

def calculate_angle(keypoint1, keypoint2, keypoint3, plane):
    global angle
    # Extract x, y, and z coordinates for each keypoint
    x1, y1, z1 = keypoint1
    x2, y2, z2 = keypoint2
    x3, y3, z3 = keypoint3

    # For test z-coordinates
    keypoint1 = x1, y1, z1*250
    keypoint2 = x2, y2, z2*250
    keypoint3 = x3, y3, z3*250

    if plane == 'front':
        # Calculate vectors between keypoints in the XY-plane
        vector1 = np.array([x1 - x2, y1 - y2, 0])
        vector2 = np.array([x3 - x2, y3 - y2, 0])
    elif plane == 'top':
        # Calculate vectors between keypoints in the XZ-plane
        vector1 = np.array([x1 - x2, 0, (z1 - z2)*250])
        vector2 = np.array([x3 - x2, 0, (z3 - z2)*250])
    elif plane == 'side':
        # Calculate vectors between keypoints in the YZ-plane
        vector1 = np.array([0, y1 - y2, (z1 - z2)*250])
        vector2 = np.array([0, y3 - y2, (z3 - z2)*250])

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
    keypoint1 = x1, y1, z1 * 250
    keypoint2 = x2, y2, z2 * 250
    keypoint3 = x3, y3, z3 * 250
    keypoint4 = x4, y4, z4 * 250
    
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