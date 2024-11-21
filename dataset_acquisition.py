import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Tuple, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING logs 
sys.stdout.flush()

class GestureDataCollector:
    """Main class for collecting and processing gesture data"""

    RAW_DATA_PATH: str = "data/raw/"
    PROCESSED_DATA_PATH: str = "data/processed/"
    SEQUENCE_LENGTH: int = 20
    IMG_SIZE: tuple = (64, 64)  # Resizing to 64x64 for model training

    def __init__(self):
        """Initialize MediaPipe Hands and create necessary directories"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        os.makedirs(self.RAW_DATA_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Preprocess a single frame using MediaPipe to extract hand landmarks and ROI.

        Args:
            frame: Input video frame

        Returns:
            - ROI of the hand (if detected), resized and normalized.
            - Frame with landmarks drawn (for display purposes).
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_roi = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Calculate bounding box for the hand
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Ensure bounding box is within bounds
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # Crop and preprocess the ROI
                hand_roi = frame[y_min:y_max, x_min:x_max]
                if hand_roi.size != 0:
                    hand_roi = cv2.resize(hand_roi, self.IMG_SIZE)
                    hand_roi = hand_roi / 255.0  # Normalize pixel values

        return hand_roi, frame

    def collect_data(self, label: str, num_sequences: int, capture_delay: int) -> None:
        """
        Collect gesture data for a specific label.

        Args:
            label: The gesture label to collect data for
            num_sequences: Number of gesture sequences to capture
            capture_delay: Delay (in seconds) between captures
        """
        label_path_processed = Path(self.PROCESSED_DATA_PATH) / label
        label_path_processed.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not accessible.")
            return

        print(f"Capturing {num_sequences} sequences for gesture: {label}")
        sequence_count = 0

        try:
            while sequence_count < num_sequences:
                frames = []
                print(f"Capturing sequence {sequence_count + 1}/{num_sequences}...")
                for frame_idx in range(self.SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame.")
                        break

                    # Preprocess the frame and get the ROI
                    roi, display_frame = self.preprocess_frame(frame)

                    if roi is not None:
                        frames.append(roi)  # Append processed frame to sequence

                    # Display the frame with landmarks and progress
                    cv2.putText(display_frame, f"Sequence {sequence_count + 1}/{num_sequences}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Capture Gesture", display_frame)

                    # Wait for delay or quit if 'q' is pressed
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        print("Exiting gesture collection.")
                        return

                if len(frames) == self.SEQUENCE_LENGTH:
                    # Save the processed sequence
                    frames = np.array(frames)
                    processed_file_path = label_path_processed / f"{label}_{sequence_count}.npy"
                    np.save(str(processed_file_path), frames)
                    print(f"Saved processed sequence: {processed_file_path}")
                    sequence_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

        print(f"Gesture '{label}' sequences saved in: {label_path_processed}")


def main():
    """Main entry point for data collection"""
    collector = GestureDataCollector()
    label = input("Enter the gesture name (e.g., 'thumbs_up'): ")
    num_sequences = int(input("Enter the number of sequences to capture (e.g., 10): "))
    capture_delay = int(input("Enter the delay between captures in seconds (e.g., 2): "))
    collector.collect_data(label, num_sequences, capture_delay)


if __name__ == "__main__":
    main()
