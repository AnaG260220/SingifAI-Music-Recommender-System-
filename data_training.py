import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam 
from sklearn.utils import shuffle

# Initialize variables and lists
initialized = False
data_size = -1
labels = []
label_dictionary = {}
class_count = 0

# Load and preprocess the data
for file in os.listdir():
    if file.endswith(".npy") and not file.startswith("labels"):
        label_name = file.split(".")[0]
        if not initialized:
            initialized = True
            X = np.load(file)
            data_size = X.shape[0]
            y = np.array([label_name] * data_size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(file)))
            y = np.concatenate((y, np.array([label_name] * data_size).reshape(-1, 1)))

        labels.append(label_name)
        label_dictionary[label_name] = class_count
        class_count += 1

# Map labels to integers
for i in range(y.shape[0]):
    y[i, 0] = label_dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert labels to one-hot encoding
y = to_categorical(y)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Create the neural network model
input_layer = Input(shape=(X.shape[1]))
hidden_layer1 = Dense(512, activation="relu")(input_layer)
hidden_layer2 = Dense(256, activation="relu")(hidden_layer1)
output_layer = Dense(y.shape[1], activation="softmax")(hidden_layer2)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50)

# Save the trained model and labels
model.save("model.h5")
np.save("labels.npy", np.array(labels))
