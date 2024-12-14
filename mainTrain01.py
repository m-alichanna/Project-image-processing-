import cv2
from keras.models import load_model
import numpy as np
import os

# Check if the model file exists
model_path = "./trainingDataTarget/model-014.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

# Load the pre-trained model
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video stream")
    exit()

# Check if the haarcascades file exists
haarcascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(haarcascade_path):
    print(f"Error: haarcascade file not found at {haarcascade_path}")
    exit()

faceDetect = cv2.CascadeClassifier(haarcascade_path)

color_dict = {0:(0,0,255), 1:(0,255,0)}  # Color for labels (Female: red, Male: green)
labels_dict = {0:"Female", 1:"Male"}  # Labels for classification

while True:
    ret, frame = video.read()  # Read video frame
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale for face detection
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)  # Detect faces in the frame

    for x, y, w, h in faces:
        # Extract face region from the grayscale image
        sub_face_img = gray[y:y+h, x:x+w]
        
        # Resize to 32x32 and normalize the image for model prediction
        resized = cv2.resize(sub_face_img, (32, 32))
        normalize = resized / 255.0  # Normalize pixel values to [0, 1]
        
        # Reshape for model input: (1, 32, 32, 1)
        reshaped = np.reshape(normalize, (1, 32, 32, 1))
        
        # Predict with the model
        try:
            result = model.predict(reshaped)
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue
        
        # Get the label with the highest probability
        label = np.argmax(result, axis=1)[0]
        print(f"Predicted label: {label} ({labels_dict[label]})")

        # Draw rectangle around face and display the label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Red rectangle for face boundary
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)  # Green/Red rectangle for gender
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], -1)  # Background for label
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame with face detection and label
    try:
        cv2.imshow("Frame", frame)
    except Exception as e:
        print(f"Error displaying frame: {e}")
        break

    # Wait for key press, exit if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Release resources and close windows
video.release()
cv2.destroyAllWindows()
