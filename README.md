# Gesture-Based Control System

The Gesture-Based Control System is an advanced application that enables touchless interaction with your computer using hand gestures. Powered by computer vision and machine learning, this system recognizes gestures in real-time and maps them to predefined actions like scrolling, clicking, adjusting volume, and more. It provides an intuitive and seamless hands-free experience, ideal for productivity, accessibility, and entertainment.

---

## Features

- **Real-Time Gesture Recognition**:
  - Uses Mediapipe's Hand Tracking to detect and analyze hand landmarks.
  - Processes gestures with high accuracy and low latency.

- **Customizable Actions**:
  - Maps gestures to specific actions such as scrolling, media control, and system brightness adjustment.
  - Supports adding new gestures and actions.

- **AI-Powered**:
  - Employs a deep learning model trained to recognize a wide variety of gestures.
  - Predictions refined using a label encoder for accuracy.

- **Visual Feedback**:
  - Displays recognized gestures on-screen with fade-in and fade-out animations.

- **System Integration**:
  - Controls the system using PyAutoGUI, keyboard simulation, and brightness adjustment tools.

- **Cross-Platform Compatibility**:
  - Runs on Windows, macOS, and Linux.

---

## Gesture-to-Action Mapping

| **Gesture**      | **Action**           |
|-------------------|----------------------|
| Thumb Up          | Increase Volume      |
| Thumb Down        | Decrease Volume      |
| Peace Sign        | Play/Pause Media     |
| Swipe Left        | Previous Track       |
| Swipe Right       | Next Track           |
| Open Hand         | Scroll Up            |
| Swipe Down        | Scroll Down          |

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/gesture-control-system.git
   cd gesture-control-system
   ```
2. Install dependencies:
 ```bash
  pip install -r requirements.txt
  ```
3. Download the pre-trained model and label encoder:
Place the model file (gesture_model.keras) and label encoder (label_encoder.pkl) in the models/ directory.
Run the application:

```bash
python gesture_control_system
```
Usage
Start the program and ensure your webcam is connected.
Perform gestures in front of the camera.
The system will recognize gestures and perform the mapped actions:
Example: Peace Sign pauses or plays media.
Keyboard Shortcut
Press q to exit the program.
Requirements
Python 3.8+
Libraries:
OpenCV
Mediapipe
TensorFlow/Keras
PyAutoGUI
Keyboard
Screen-Brightness-Control (for brightness adjustment)
Install all dependencies via pip install -r requirements.txt.

File Structure
gesture-control-system/
│
├── models/
│   ├── gesture_model.keras    # Trained gesture recognition model
│   ├── label_encoder.pkl      # Label encoder for gesture classification
│
├── gesture_control_system                    # Main script
├── requirements.txt           # Dependency list
└── README.md                  # Project documentation
Potential Enhancements
Voice Command Integration: Combine gestures with voice for hybrid control.
Multi-Hand Gestures: Enable more complex two-hand gesture controls.
AR/VR Support: Extend functionality to virtual and augmented reality platforms.
Custom Gesture Training: Allow users to define their own gestures and actions.
License
This project is licensed under the MIT License.

Acknowledgments
Mediapipe for its powerful hand-tracking API.
Open-source contributors for the Python libraries used in this project.
Contact
For questions or support, please contact:

Moon9T
Email: thyrook@proton.me
GitHub: Moon9T
