import cv2
import numpy as np
from keras.models import load_model

# Error handling for loading the model
try:
    model = load_model("./trainingDataTarget/model-014.h5")  # Ensure the path is correct
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None to indicate an error

# Initialize the face detector using Haar Cascade Classifier
try:
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if faceDetect.empty():
        raise ValueError("Haar Cascade classifier could not be loaded.")
    print("Haar Cascade loaded successfully")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    faceDetect = None  # Set faceDetect to None to indicate an error

# Define color and label dictionaries for gender classification
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}  # Red for Female, Green for Male
labels_dict = {0: "Female", 1: "Male"}

class Video(object):
    def __init__(self):
        # Initialize the video capture (use the default webcam)
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Error: Could not open video source.")
            raise Exception("Could not open video source")

    def __del__(self):
        # Release the video capture when the object is deleted
        self.video.release()

    def get_frame(self):
        # Capture a frame from the video feed
        ret, frame = self.video.read()
        if not ret:
            print("Failed to capture frame")
            return None

        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if faceDetect is None:
            return frame  # Return the frame without face detection if the classifier is not loaded

        # Detect faces in the grayscale image
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            # Draw rectangles and lines around the detected face
            x1, y1 = x + w, y + h
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)
            cv2.line(frame, (x, y), (x + 30, y), (255, 0, 255), 6)
            cv2.line(frame, (x, y), (x, y + 30), (255, 0, 255), 6)
            cv2.line(frame, (x1, y), (x1 - 30, y), (255, 0, 255), 6)
            cv2.line(frame, (x1, y), (x1, y + 30), (255, 0, 255), 6)
            cv2.line(frame, (x, y1), (x + 30, y1), (255, 0, 255), 6)
            cv2.line(frame, (x, y1), (x, y1 - 30), (255, 0, 255), 6)
            cv2.line(frame, (x1, y1), (x1 - 30, y1), (255, 0, 255), 6)
            cv2.line(frame, (x1, y1), (x1, y1 - 30), (255, 0, 255), 6)

            # Extract the face region from the grayscale image for prediction
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (32, 32))
            normalized = resized / 255.0  # Normalize the pixel values to [0, 1]
            reshaped = np.reshape(normalized, (1, 32, 32, 1))  # Reshape to match model input

            try:
                if model:
                    # Predict the label (gender) using the model
                    result = model.predict(reshaped)
                    label = np.argmax(result, axis=1)[0]  # Get the predicted label

                    # Draw a background rectangle and label the face with gender
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], -1)
                    cv2.putText(frame, labels_dict[label], (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error during prediction: {e}")

        # Return the processed frame directly as a NumPy array (not JPEG-encoded)
        return frame

# Initialize video object and start capturing frames
try:
    video_stream = Video()
except Exception as e:
    print(f"Error initializing video stream: {e}")
    video_stream = None

if video_stream:
    while True:
        # Get the frame from the video stream
        frame = video_stream.get_frame()
        if frame is None:
            print("No frame to process.")
            break

        # Display the frame with face detection and gender label
        try:
            cv2.imshow("Camera Feed", frame)  # Directly pass the raw frame here
        except cv2.error as e:
            print(f"Error displaying frame: {e}")

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close all OpenCV windows
    try:
        video_stream.__del__()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    cv2.destroyAllWindows()
