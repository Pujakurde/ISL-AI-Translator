'''import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Load hand landmark data
def load_data(directory):
    data = []
    labels = []
    label_mapping = {}
    label_index = 0
    for subfolder in os.listdir(directory):
        if subfolder not in label_mapping:
            label_mapping[subfolder] = label_index
            label_index += 1
        subfolder_path = os.path.join(directory, subfolder)
        print(f"Processing subfolder: {subfolder}")
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                landmarks = []
                for line in lines:
                    parts = line.split(',')
                    x_part = [part for part in parts if 'x=' in part]
                    y_part = [part for part in parts if 'y=' in part]
                    if x_part and y_part:
                        x = float(x_part[0].split('x=')[1])
                        y = float(y_part[0].split('y=')[1])
                        landmarks.append(x)
                        landmarks.append(y)
                if len(landmarks) == 42: # Ensure each sample has 42 features
                    data.append(landmarks)
                    labels.append(label_mapping[subfolder])
    print(f"Loaded {len(data)} hand landmarks")
    return np.array(data), np.array(labels)

# Load data
print("Loading data...")
data, labels = load_data('onehand_landmarks')
print("Data loaded.")

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
print("Data split.")

# Define neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(42,)))
model.add(Dropout(0.2))
model.add(Dense(len(set(labels)), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
print("Model trained.")

model.save('onehand_gesture_model.h5')
print("Model saved.")'''
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATASET_DIR = "dataset/Indian/onehand"
MODEL_SAVE = "models/isl_onehand_model.h5"
LABELS_SAVE = "models/isl_onehand_labels.json"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs("models", exist_ok=True)

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = len(train_gen.class_indices)

# Model
model = models.Sequential([
    layers.Input(shape=(64,64,3)),

    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
model.save(MODEL_SAVE)

# Save labels
with open(LABELS_SAVE, "w") as f:
    json.dump(train_gen.class_indices, f)

print("One-hand CNN model saved")