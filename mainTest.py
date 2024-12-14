import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model("./trainingDataTarget/model-014.h5")

# Initialize the face detector
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define color and label dictionaries
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
labels_dict = {0: "Female", 1: "Male"}

class Video(object):
    def __init__(self):
        # Initialize the video capture
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Release the video capture when the object is deleted
        self.video.release()

    def get_frame(self):
        # Read a frame from the video feed
        ret, frame = self.video.read()
        if not ret:
            return None
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            x1, y1 = x + w, y + h

            # Draw rectangles and decorative lines around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)
            cv2.line(frame, (x, y), (x + 30, y), (255, 0, 255), 6)
            cv2.line(frame, (x, y), (x, y + 30), (255, 0, 255), 6)

            cv2.line(frame, (x1, y), (x1 - 30, y), (255, 0, 255), 6)
            cv2.line(frame, (x1, y), (x1, y + 30), (255, 0, 255), 6)

            cv2.line(frame, (x, y1), (x + 30, y1), (255, 0, 255), 6)
            cv2.line(frame, (x, y1), (x, y1 - 30), (255, 0, 255), 6)

            cv2.line(frame, (x1, y1), (x1 - 30, y1), (255, 0, 255), 6)
            cv2.line(frame, (x1, y1), (x1, y1 - 30), (255, 0, 255), 6)

            sub_face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(sub_face, (32, 32))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 32, 32, 1))
            result = model.predict(reshaped_face)

            label = np.argmax(result, axis=1)[0]
            label_name = labels_dict[label]
            print(f"Predicted label: {label_name}")

            # Display label with rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.putText(frame, label_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

# Initialize the video stream and process frames
video_stream = Video()

while True:
    frame = video_stream.get_frame()
    if frame is None:
        break
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
