'''import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Define constants
NUM_HAND_LANDMARKS = 21
NUM_CLASSES = len(os.listdir('numerical_landmarks'))

# ADD THIS FUNCTION
def get_max_landmarks(directory):
    max_lm = 0
    for subfolder in os.listdir(directory):
        folder_path = os.path.join(directory, subfolder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()[1:]
                if len(lines) > max_lm:
                    max_lm = len(lines)
    return max_lm

# ADD THIS LINE
max_landmarks = get_max_landmarks('numerical_landmarks')

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
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()[1:]  # skip the first line
                landmarks = []
                for line in lines:
                    parts = line.split(',')
                    if "x=" in parts[0] and "y=" in parts[1]:
                        x = float(parts[0].split('=')[1])
                        y = float(parts[1].split('=')[1])
                        landmarks.append(x)
                        landmarks.append(y)

                # NOW THIS WORKS
                landmarks += [0] * (max_landmarks * 2 - len(landmarks))

                data.append(landmarks)
                labels.append(label_mapping[subfolder])

    return np.array(data), np.array(labels)


# Load data
data, labels = load_data('numerical_landmarks')

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(max_landmarks * 2,)))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
model.save('numericalhand_gesture_model2.0.h5')'''

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras impo
MODEL_SAVE  = "models/isl_number_model.h5"
LABELS_SAVE = "models/isl_number_labels.json"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs("models", exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = len(train_gen.class_indices)

model = models.Sequential([
    layers.Input(shape=(64,64,3)),

    layers.Conv2D(32,(3,3),activation="relu",padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128,(3,3),activation="relu",padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256,activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES,activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(MODEL_SAVE, save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# SAVE LABELS
with open(LABELS_SAVE, "w") as f:
    json.dump(train_gen.class_indices, f)

print("Number model saved")
