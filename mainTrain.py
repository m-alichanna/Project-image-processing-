import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# Path and class setup
datapath = "Dataset"
classes = os.listdir(datapath)
labels = [i for i in range(len(classes))]
label_dict = dict(zip(classes, labels))
print(label_dict)

# Load the data
img_size = 32
data = []
target = []
facedata = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

# Loop through each image in the dataset
for category in classes:
    folder_path = os.path.join(datapath, category)
    img_names = os.listdir(folder_path)
    
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = cascade.detectMultiScale(gray)
        
        try:
            for x, y, w, h in faces:
                sub_face = img[y:y + h, x:x + w]
                resized = cv2.resize(sub_face, (img_size, img_size))
                gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Convert resized face to grayscale
                data.append(gray_resized)
                target.append(label_dict[category])
                
        except Exception as e:
            print("Exception:", e)

# Convert data and target to numpy arrays
data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

# One-hot encode the target
new_target = np_utils.to_categorical(target)

# Save the processed data and target
np.save("./trainingDataTarget/data", data)
np.save("./trainingDataTarget/target", new_target)

# Load data
data = np.load("./trainingDataTarget/data.npy")
target = np.load("./trainingDataTarget/target.npy")

# Split data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)
print(train_data.shape)

# CNN Model
noOfFilters = 64
sizeOfFilters1 = (3, 3)
sizeOfFilters2 = (3, 3)
sizeOfPool = (2, 2)
noOfNode = 64

# Define the model
model = Sequential()
model.add(Conv2D(32, sizeOfFilters1, input_shape=data.shape[1:], activation="relu"))
model.add(Conv2D(32, sizeOfFilters1, activation="relu"))
model.add(MaxPooling2D(pool_size=sizeOfPool))

model.add(Conv2D(64, sizeOfFilters2, activation="relu"))
model.add(Conv2D(64, sizeOfFilters2, activation="relu"))
model.add(MaxPooling2D(pool_size=sizeOfFilters2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(noOfNode, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

# Model Training with ModelCheckpoint to save in .h5 format
checkpoint = ModelCheckpoint("./trainingDataTarget/model-{epoch:03d}.h5", monitor="val_loss", verbose=0, save_best_only=True, mode="min")
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint], validation_split=0.2)
