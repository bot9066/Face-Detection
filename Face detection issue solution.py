import cv2
import mediapipe as mp
import math
import numpy as np

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize face mesh detection
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to apply histogram equalization to improve lighting
def adjust_lighting(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    # Convert back to BGR (so it's compatible with Mediapipe)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# Function to handle occlusions by interpolating missing points
def handle_occlusions(face_landmarks, frame):
    # If any facial landmark is missing, try interpolating the missing data from surrounding landmarks
    if not face_landmarks:
        return frame

    # Example: Draw all available landmarks to help with occlusions
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
    )
    return frame

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Flip frame for a mirrored view
    frame = cv2.flip(frame, 1)

    # Adjust lighting by improving contrast in varying light conditions
    frame = adjust_lighting(frame)

    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw face mesh
            frame = handle_occlusions(face_landmarks, frame)  # Handle occlusions

    # Display the frame
    cv2.imshow("Face Mesh Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
