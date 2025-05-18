# Face-Detection
# 🧠 Real-Time Face Mesh Detection with Mediapipe

This project implements a real-time face mesh detection system using Mediapipe and OpenCV, enhanced with:

- ✅ Automatic lighting adjustment using histogram equalization
- ✅ Handling of face occlusions (e.g., partial face visibility)
- ✅ Real-time webcam feed processing

---

## 📸 Features

- Detects 468 facial landmarks in real-time
- Enhances visibility in poor lighting conditions
- Attempts to display available facial landmarks even with occlusions
- Real-time performance with Python and OpenCV

---

## 🔧 Requirements

Install the following Python libraries before running the code:


pip install opencv-python mediapipe numpy

## 📁 Project Files

Face-Detect ion/
├── face-Detect.py      # Main script for face mesh detection
├── README.md           # Project documentation.    

## 🧠 How It Works

Lighting Adjustment
Poor lighting affects facial detection accuracy. This project uses histogram equalization to enhance contrast:

def adjust_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
## Occlusion Handling

If parts of the face are blocked, landmarks that are still detected will be drawn:

python
Copy
Edit
def handle_occlusions(face_landmarks, frame):
    if not face_landmarks:
        return frame
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
    )
    return frame
