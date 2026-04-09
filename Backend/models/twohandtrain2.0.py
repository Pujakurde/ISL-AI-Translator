'''import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

DATASET_DIR = "dataset/two_hand"
MODEL_SAVE  = "models/isl_twohand_model.h5"
LABELS_SAVE = "models/isl_twohand_labels.json"
#Load data

X = np.load('X.npy')  # shape (num_samples, num_landmarks*2)
y = np.load('y.npy')  # shape (num_samples,) with letters 'A', 'B', etc.


le = LabelEncoder()
y_int = le.fit_transform(y)  # letters → integers
num_classes = len(np.unique(y_int))
y_cat = to_categorical(y_int, num_classes=num_classes)

# Save label encoder for decoding predictions later
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)


# Normalize landmarks

def normalize_landmarks(X):
    X_norm = []
    for sample in X:
        lm = sample.reshape(-1, 2)
        center = np.mean(lm, axis=0)
        lm -= center
        max_val = np.max(np.linalg.norm(lm, axis=1))
        if max_val > 0:
            lm /= max_val
        X_norm.append(lm.flatten())
    return np.array(X_norm)

X = normalize_landmarks(X)

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_int
)


# Compute class weights

class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=y_int
)
class_weights = dict(enumerate(class_weights_values))


#Build deeper model

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train model

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

#  Save the trained model
model.save('improved_hand_gesture_model.h5')
print("Model trained and saved successfully!")
'''
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATASET_DIR = "dataset/Indian/twohand"
MODEL_SAVE = "models/isl_twohand_model.h5"
LABELS_SAVE = "models/isl_twohand_labels.json"

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

print("Two-hand CNN model saved")