from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

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

# Color and label dictionaries for gender classification
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}  # Red for Female, Green for Male
labels_dict = {0: "Female", 1: "Male"}

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Error: Could not open video source.")
            raise Exception("Could not open video source")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            print("Failed to capture frame")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if faceDetect is not None:
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            for x, y, w, h in faces:
                x1, y1 = x + w, y + h
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

                sub_face_img = gray[y:y + h, x:x + w]
                resized = cv2.resize(sub_face_img, (32, 32))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 32, 32, 1))

                try:
                    if model:
                        result = model.predict(reshaped)
                        label = np.argmax(result, axis=1)[0]

                        cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], -1)
                        cv2.putText(frame, labels_dict[label], (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Error during prediction: {e}")

        return frame

@app.route("/")
def index():
    return render_template("index.html")

def gen(video_stream):
    while True:
        frame = video_stream.get_frame()
        if frame is None:
            break

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video')
def video():
    video_stream = Video()
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
