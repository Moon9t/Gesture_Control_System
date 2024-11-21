import os
import numpy as np
import cv2
from keras.api.models import Sequential
from keras.api.layers import (ConvLSTM2D, 
            TimeDistributed, Conv2D, Conv3D, 
            MaxPooling2D, MaxPooling3D, 
            Flatten, Dense, Dropout, 
            BatchNormalization,Input)
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Path to the dataset and model save path
DATASET_PATH = "data/processed/"
MODEL_SAVE_PATH = "models/gesture_model.keras"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# Parameters
IMG_SIZE = (128, 128)  # Resize images
SEQUENCE_LENGTH = 10  # Number of frames per gesture
BATCH_SIZE = 32
EPOCHS = 1500
LEARNING_RATE = 0.001

# Load Data
def load_data(dataset_path):
    gestures = []
    labels = []
    
    for root, dirs, files in os.walk(dataset_path):
        npy_files = [f for f in files if f.endswith('.npy')]
        if not npy_files:
            continue
            
        label = os.path.basename(root)
        frames = []
        
        for frame_file in sorted(npy_files)[:SEQUENCE_LENGTH]:
            frame_path = os.path.join(root, frame_file)
            try:
                # Load frame data
                img = np.load(frame_path)
                print(f"Loaded {frame_path} with shape {img.shape}")
                
                # Handle grayscale conversion based on number of channels
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                    img = img.astype(np.float32)
                
                # Ensure 3D shape (height, width, 1)
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                
                frames.append(img)
                
            except Exception as e:
                print(f"Error loading {frame_path}: {str(e)}")
                continue
                
        if len(frames) == SEQUENCE_LENGTH:
            frames_array = np.array(frames)
            gestures.append(frames_array)
            labels.append(label)
        else:
            print(f"Skipped gesture {label} due to insufficient frames: {len(frames)}")
            
    if not gestures:
        raise ValueError("No valid sequences found in the dataset")
        
    return np.array(gestures), np.array(labels)
# Load and preprocess data
print("Loading and preprocessing data...")
gestures, labels = load_data(DATASET_PATH)
# After loading data
print("Initial data shapes:")
print("Gestures shape:", gestures.shape)
print("Labels shape:", labels.shape)

# Get unique labels and their counts
unique_labels = np.unique(labels)
print("Unique labels:", unique_labels)
print("Number of unique labels:", len(unique_labels))

# Ensure equal number of samples
n_samples = min(len(gestures), len(labels))
gestures = gestures[:n_samples]
labels = labels[:n_samples]

# After loading and preprocessing data
print("Initial gestures shape:", gestures.shape)

# Reshape gestures to match model's expected shape (None, 10, 64, 64, 3)
# Remove the extra dimension of 20 by taking the first frame of each sequence
gestures = gestures[:, :, 0, :, :, :]

print("Reshaped gestures shape:", gestures.shape)

# Normalize if not already done
gestures = gestures.astype('float32') / 255.0

# Encode labels
label_encoder = LabelBinarizer()
encoded_labels = label_encoder.fit_transform(labels)

# Define the model
model = Sequential()

# Add layers to the model
model.add(Input(shape=(10, 64, 64, 3)))  # Example input shape
model.add(Conv3D(16, (3, 3, 3), activation='relu', input_shape=(10, 64, 64, 3)))  # Example input shape
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
from keras.api.layers import Dense
from keras.api.regularizers import l2

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))  # Example with L2 regularization# Get number of classes from encoded labels
num_classes = len(label_encoder.classes_)

# Update the final Dense layer to match number of classes
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Print model summary
model.summary()

# Save the label encoder
with open(LABEL_ENCODER_PATH, 'wb') as file:
    pickle.dump(label_encoder, file)

# Split with correctly shaped data
X_train, X_test, y_train, y_test = train_test_split(
    gestures, 
    encoded_labels,
    test_size=0.2, 
    random_state=42
)

print("Training shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

history = model.fit(
X_train, y_train,
validation_data=(X_test, y_test),
batch_size=BATCH_SIZE,
epochs=EPOCHS,
verbose=1
)

# Save the label encoder
with open(LABEL_ENCODER_PATH, 'wb') as file:
    pickle.dump(label_encoder, file)
# Data Augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Print debug information
print("IMG_SIZE:", IMG_SIZE)
print("Initial gestures shape:", gestures.shape)

# Apply augmentation frame by frame
augmented_gestures = []
# Process each frame
for gesture_idx, gesture in enumerate(gestures):
    augmented_sequence = []
    for frame_idx, frame in enumerate(gesture):
        print(f"Processing gesture {gesture_idx}, frame {frame_idx}")
        
        # Skip invalid frames
        if frame.size == 0 or frame.ndim < 2:
            print(f"Invalid frame at gesture {gesture_idx}, frame {frame_idx}")
            continue
        
        # Ensure target size is valid
        target_size = (IMG_SIZE[1], IMG_SIZE[0])  # cv2 expects (width, height)
        
        # Convert frame to uint8 and resize
        try:
            frame = (frame * 255).astype(np.uint8)
            frame_resized = cv2.resize(frame, target_size)
            
            # Add a channel if grayscale
            if frame_resized.ndim == 2:
                frame_resized = np.expand_dims(frame_resized, axis=-1)
            
            # Convert to RGB
            frame_rgb = np.repeat(frame_resized, 3, axis=-1)
            
            # Apply augmentation
            augmented_frame = data_augmentation.random_transform(frame_rgb)
            augmented_sequence.append(augmented_frame)
        except Exception as e:
            print(f"Error processing frame {frame_idx} in gesture {gesture_idx}: {e}")
            continue
    
    if augmented_sequence:
        augmented_gestures.append(augmented_sequence)


# Convert to numpy array
gestures = np.array(augmented_gestures)
gestures = gestures.astype('float32') / 255.0
print("Final gestures shape:", gestures.shape)
print("Building model...")
model = Sequential([
   # Shape the input to combine sequence and frames dimensions
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(BatchNormalization()),

    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(BatchNormalization()),

    TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(BatchNormalization()),

    # ConvLSTM2D layer
    ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False),
    BatchNormalization(),
    Dropout(0.5),

    # Dense layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax'),
    Dropout(0.5)  # Add this after Dense layers
])

# Compile Model
print("Compiling model...")
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Defining callbacks
callbacks = [
    ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    TensorBoard(
        log_dir='logs/fit',
        histogram_freq=1
    )
]

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train Model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# Save Model
# Update the file extension
MODEL_SAVE_PATH = "models/gesture_model.keras"
print(f"Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)

# Save model training history
with open('model_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Plot Model Evaluation
def plot_training_history(history):
    fig = plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.show()
# Plot the training history
plot_training_history(history)
# Print a message to indicate training completion
print("Training complete!")




