import cv2
import numpy as np
import mediapipe as mp
from keras.api.models import load_model
import pickle
import time
import pyautogui
import keyboard
import json
import os
import logging
import pyttsx3
import screen_brightness_control as sbc

# Paths
MODEL_PATH = "models/gesture_model.keras"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
CONFIG_PATH = "gesture_config.json"

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load Model and Label Encoder
model = load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as file:
    label_encoder = pickle.load(file)

# Parameters
IMG_SIZE = (64, 64)
SEQUENCE_LENGTH = 20
PREDICTION_THRESHOLD = 0.8

# Load gesture-to-action mapping
def load_gesture_action_map(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        return json.load(file)["gestures"]

gesture_action_map = load_gesture_action_map()

# Setup logging
logging.basicConfig(filename='gesture_logs.txt', level=logging.INFO)

# Action handler functions
def mouse_click():
    pyautogui.click()

def scroll_up():
    pyautogui.scroll(10)

def scroll_down():
    pyautogui.scroll(-10)

def volume_up():
    keyboard.press_and_release('volume up')

def volume_down():
    keyboard.press_and_release('volume down')

def pause_play():
    keyboard.press_and_release('space')

def previous_track():
    keyboard.press_and_release('media previous track')

def next_track():
    keyboard.press_and_release('media next track')

def increase_brightness():
    sbc.set_brightness('+10')

def decrease_brightness():
    sbc.set_brightness('-10')

def open_app(app_name):
    if app_name == "chrome":
        os.system("start chrome")
    elif app_name == "notepad":
        os.system("start notepad")

def perform_action(gesture):
    """
    Perform the action associated with the recognized gesture.
    """
    action = gesture_action_map.get(gesture)
    if action:
        print(f"Performing action: {action}")
        eval(action)  # Safely execute the mapped function

def log_gesture(gesture):
    """
    Log recognized gestures for analytics.
    """
    logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Recognized Gesture: {gesture}")
    print(f"Logged Gesture: {gesture}")

def provide_feedback(gesture):
    """
    Provide audio feedback for recognized gestures.
    """
    engine = pyttsx3.init()
    engine.say(f"Gesture recognized: {gesture}")
    engine.runAndWait()

def preprocess_frame(frame):
    """
    Preprocess a single frame using Mediapipe to extract hand landmarks.
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    blank_image = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * IMG_SIZE[1])
                y = int(landmark.y * IMG_SIZE[0])
                cv2.circle(blank_image, (x, y), 2, 255, -1)
    return blank_image

def predict_gesture(sequence):
    """
    Predict the gesture from a sequence of frames.
    """
    sequence = np.expand_dims(sequence, axis=0)
    predictions = model.predict(sequence)[0]
    max_prob = np.max(predictions)
    if max_prob > PREDICTION_THRESHOLD:
        return label_encoder.classes_[np.argmax(predictions)]
    return None

def save_training_data(gesture_name, frames):
    """
    Save frames for training a new gesture.
    """
    path = f"training_data/{gesture_name}/"
    os.makedirs(path, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{path}/{i}.png", frame)
    print(f"Saved {len(frames)} frames for gesture '{gesture_name}'.")

def train_new_gesture(cap):
    """
    Train a new gesture using webcam data.
    """
    print("Starting training mode. Press 'q' to exit.")
    frames = []
    gesture_name = input("Enter the gesture name: ")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
        cv2.imshow("Training Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    save_training_data(gesture_name, frames)
    cap.release()
    cv2.destroyAllWindows()

# Main application loop
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sequence = []
    last_time = time.time()
    debounce_time = 1.0
    silent_mode = False
    current_gesture = None

    print("Starting gesture recognition. Press 'q' to quit or 't' to train a new gesture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected.")
            break

        preprocessed_frame = preprocess_frame(frame)
        sequence.append(preprocessed_frame)

        if len(sequence) == SEQUENCE_LENGTH:
            gesture = predict_gesture(np.array(sequence))
            if gesture and (time.time() - last_time > debounce_time):
                current_gesture = gesture
                print(f"Recognized Gesture: {gesture}")
                if not silent_mode:
                    perform_action(current_gesture)
                    log_gesture(current_gesture)
                    provide_feedback(current_gesture)
                last_time = time.time()
            sequence.pop(0)

        cv2.imshow("Gesture Control System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            train_new_gesture(cap)
        elif key == ord('s'):
            silent_mode = not silent_mode
            print(f"Silent Mode: {'ON' if silent_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting gesture recognition.")

if __name__ == "__main__":
    main()
