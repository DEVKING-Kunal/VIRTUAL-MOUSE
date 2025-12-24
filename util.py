import numpy as np

def get_angle(a, b, c):
    """
    Calculates the angle at point 'b' given three points a, b, c.
    Returns angle in degrees.
    """
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_distance(landmark_list):
    """
    Calculates Euclidean distance between two landmarks.
    Expects landmark_list to be [(x1, y1), (x2, y2)]
    """
    if len(landmark_list) < 2:
        return 0
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    # MediaPipe coords are 0-1. Max distance is approx 1.41.
    # We map 0-1 to 0-1000 to keep your original logic scale.
    return np.interp(L, [0, 1], [0, 1000])