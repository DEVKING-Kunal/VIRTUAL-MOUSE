import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import util
import pyautogui
from pynput.mouse import Button, Controller
import math
import sys
import numpy as np

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

mouse = Controller()
mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

# --- CONFIGURATION ---
ROI_SIZE = 420
CLICK_COOLDOWN_LIMIT = 5
ZOOM_COOLDOWN_LIMIT = 10
ZOOM_THRESHOLD = 0.04
SCROLL_SENSITIVITY = 15  # Pixels per frame move to trigger scroll

# --- KALMAN FILTER SETUP ---
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                 [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                [0, 1, 0, 1], 
                                [0, 0, 1, 0], 
                                [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.005 
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

# Global State Variables
click_cooldown = 0
zoom_cooldown = 0
prev_zoom_dist = None
dragging = False
prev_pinky_y = None  # For scrolling

def find_finger_tip(landmarks):
    return landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

def find_index_base(landmarks):
    # Used for tracking movement during a "Fist" (Drag)
    return landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

def get_landmarks_list(hand_landmarks):
    return [(lm.x, lm.y) for lm in hand_landmarks.landmark]

def move_mouse_kalman(tracking_point, frame, sensitivity=1.0):
    """
    Uses Kalman Filter to smooth mouse movement.
    Supports 'sensitivity' for Sniper Mode.
    """
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # Region of Interest
    left = cx - ROI_SIZE // 2
    right = cx + ROI_SIZE // 2
    top = cy - ROI_SIZE // 2
    bottom = cy + ROI_SIZE // 2

    # Map hand coordinates to ROI
    fx = int(tracking_point.x * w)
    fy = int(tracking_point.y * h)

    # Clamp to box
    fx = max(left, min(right, fx))
    fy = max(top, min(bottom, fy))

    # Normalize to 0.0 - 1.0 within the box
    nx = (fx - left) / ROI_SIZE
    ny = (fy - top) / ROI_SIZE

    # Target screen coordinates
    target_x = nx * screen_width
    target_y = ny * screen_height

    # Update Kalman Filter
    measurement = np.array([[np.float32(target_x)], [np.float32(target_y)]])
    kf.correct(measurement)
    prediction = kf.predict()
    
    # Get smoothed coordinates
    smooth_x, smooth_y = prediction[0][0], prediction[1][0]

    # Apply Sensitivity (Sniper Mode)
    if sensitivity < 1.0:
        # We blend current pos with new pos heavily favoring current pos
        current_x, current_y = pyautogui.position()
        final_x = current_x + (smooth_x - current_x) * sensitivity
        final_y = current_y + (smooth_y - current_y) * sensitivity
    else:
        final_x, final_y = smooth_x, smooth_y

    try:
        pyautogui.moveTo(final_x, final_y)
    except pyautogui.FailSafeException:
        pass

# --- GESTURE DEFINITIONS ---

def is_fist(l):
    # All 4 fingers bent
    return util.get_angle(l[5], l[6], l[8]) < 90 and \
           util.get_angle(l[9], l[10], l[12]) < 90 and \
           util.get_angle(l[13], l[14], l[16]) < 90 and \
           util.get_angle(l[17], l[18], l[20]) < 90

def is_pinky_open(l):
    # Pinky Straight, others bent
    return util.get_angle(l[17], l[18], l[20]) > 150 and \
           util.get_angle(l[5], l[6], l[8]) < 90 and \
           util.get_angle(l[9], l[10], l[12]) < 90 and \
           util.get_angle(l[13], l[14], l[16]) < 90

def is_peace_sign(l):
    # Index & Middle Straight (Sniper Mode)
    # Ring & Pinky Bent
    return util.get_angle(l[5], l[6], l[8]) > 150 and \
           util.get_angle(l[9], l[10], l[12]) > 150 and \
           util.get_angle(l[13], l[14], l[16]) < 90 and \
           util.get_angle(l[17], l[18], l[20]) < 90

def is_left_click(l):
    return util.get_angle(l[5], l[6], l[8]) < 90 and \
           util.get_angle(l[9], l[10], l[12]) > 100

def is_right_click(l):
    return util.get_angle(l[5], l[6], l[8]) > 100 and \
           util.get_angle(l[9], l[10], l[12]) < 90

def is_double_click(l):
    return util.get_angle(l[5], l[6], l[8]) < 90 and \
           util.get_angle(l[9], l[10], l[12]) < 90

def is_screen_shot(l, d):
    return util.get_angle(l[5], l[6], l[8]) < 60 and \
           util.get_angle(l[9], l[10], l[12]) < 60 and d < 60

def is_exit_gesture(l):
    return util.get_angle(l[5], l[6], l[8]) > 150 and \
           util.get_angle(l[9], l[10], l[12]) > 150 and \
           util.get_angle(l[13], l[14], l[16]) > 150 and \
           util.get_angle(l[17], l[18], l[20]) > 150

def handle_zoom(landmarks_1, landmarks_2, frame):
    global prev_zoom_dist, zoom_cooldown

    idx1 = landmarks_1[8]
    idx2 = landmarks_2[8]
    curr_dist = math.hypot(idx1[0] - idx2[0], idx1[1] - idx2[1])
    
    h, w, _ = frame.shape
    p1 = (int(idx1[0] * w), int(idx1[1] * h))
    p2 = (int(idx2[0] * w), int(idx2[1] * h))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)

    if prev_zoom_dist is None:
        prev_zoom_dist = curr_dist
        return

    if zoom_cooldown > 0:
        zoom_cooldown -= 1
        return

    delta = curr_dist - prev_zoom_dist

    if abs(delta) > ZOOM_THRESHOLD:
        if delta > 0:
            pyautogui.hotkey('ctrl', '+')
            cv2.putText(frame, "Zoom In", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            pyautogui.hotkey('ctrl', '-')
            cv2.putText(frame, "Zoom Out", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        zoom_cooldown = ZOOM_COOLDOWN_LIMIT
        prev_zoom_dist = curr_dist

def handle_scroll(landmarks, frame):
    global prev_pinky_y
    
    current_pinky_y = landmarks[20][1] # Pinky Tip Y
    
    if prev_pinky_y is not None:
        delta = prev_pinky_y - current_pinky_y
        # Sensitivity check
        if abs(delta) > 0.02: 
            scroll_amount = int(delta * 80) # Scale factor
            pyautogui.scroll(scroll_amount)
            direction = "Scroll Up" if delta > 0 else "Scroll Down"
            cv2.putText(frame, direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    prev_pinky_y = current_pinky_y

def detect_gestures(frame, results):
    global click_cooldown, dragging, prev_pinky_y

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    
    cv2.rectangle(frame, (cx - ROI_SIZE // 2, cy - ROI_SIZE // 2), 
                  (cx + ROI_SIZE // 2, cy + ROI_SIZE // 2), (0, 200, 0), 2)

    hand_landmarks_0 = results.multi_hand_landmarks[0]
    landmarks_list_0 = get_landmarks_list(hand_landmarks_0)
    
    # --- 1. ZOOM CHECK (2 Hands) ---
    if len(results.multi_hand_landmarks) == 2:
        hand_landmarks_1 = results.multi_hand_landmarks[1]
        landmarks_list_1 = get_landmarks_list(hand_landmarks_1)
        handle_zoom(landmarks_list_0, landmarks_list_1, frame)
    else:
        global prev_zoom_dist
        prev_zoom_dist = None

    # --- 2. SINGLE HAND GESTURES ---
    
    # EXIT (Priority 1)
    if is_exit_gesture(landmarks_list_0):
        cv2.putText(frame, "EXITING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(500)
        sys.exit(0)

    # SCROLL (Priority 2)
    # Pinky Open, Others Closed
    if is_pinky_open(landmarks_list_0):
        handle_scroll(landmarks_list_0, frame)
        return # Skip moving mouse while scrolling
    else:
        prev_pinky_y = None

    # DRAG & DROP (Priority 3)
    if is_fist(landmarks_list_0):
        if not dragging:
            pyautogui.mouseDown()
            dragging = True
            cv2.putText(frame, "Drag Start", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Move mouse while dragging (Using Index Base/Knuckles for stability)
        index_base = find_index_base(hand_landmarks_0.landmark)
        move_mouse_kalman(index_base, frame)
        cv2.putText(frame, "Dragging...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return # Skip other gestures

    elif dragging:
        # If we were dragging but fist is gone -> Drop
        pyautogui.mouseUp()
        dragging = False
        cv2.putText(frame, "Drop", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # SNIPER MODE (Priority 4)
    # Peace Sign (Index + Middle Up) -> Move Slow
    if is_peace_sign(landmarks_list_0):
        index_tip = find_finger_tip(hand_landmarks_0.landmark)
        move_mouse_kalman(index_tip, frame, sensitivity=0.2) # 20% Speed
        cv2.putText(frame, "SNIPER MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return # Skip standard move/click

    # STANDARD MOUSE MOVE
    index_tip = find_finger_tip(hand_landmarks_0.landmark)
    thumb_index_dist = util.get_distance([landmarks_list_0[4], landmarks_list_0[5]])
    is_pointing = util.get_angle(landmarks_list_0[5], landmarks_list_0[6], landmarks_list_0[8]) > 90

    if thumb_index_dist < 60 and is_pointing:
        move_mouse_kalman(index_tip, frame)
        cv2.putText(frame, "Moving", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Reset Kalman prediction if stopped to prevent drift
        kf.statePost = np.array([[index_tip.x * screen_width], [index_tip.y * screen_height], [0], [0]], np.float32)

    # CLICKS (Standard)
    if click_cooldown > 0:
        click_cooldown -= 1
    else:
        if is_screen_shot(landmarks_list_0, thumb_index_dist):
            cv2.putText(frame, "Screenshot!", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            pyautogui.screenshot("screenshot.png")
            click_cooldown = 20
        elif is_double_click(landmarks_list_0):
            cv2.putText(frame, "Double Click", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            mouse.click(Button.left, 2)
            click_cooldown = 15
        elif is_left_click(landmarks_list_0):
            cv2.putText(frame, "Left Click", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            mouse.click(Button.left)
            click_cooldown = 8
        elif is_right_click(landmarks_list_0):
            cv2.putText(frame, "Right Click", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mouse.click(Button.right)
            click_cooldown = 8

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2 
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frameRGB)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                detect_gestures(frame, results)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()