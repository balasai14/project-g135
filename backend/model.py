import os
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths and image size
image_size = (224, 224)  # ResNet50 input size
save_folder = 'backend\\final\\face_data'  # Folder with your dataset
batch_size = 32

# Step 1: Load and preprocess the data
def load_data():
    data = []
    labels = []
    label_dict = {}
    label_idx = 0

    for person_name in os.listdir(save_folder):
        person_folder = os.path.join(save_folder, person_name)
        if os.path.isdir(person_folder):
            label_dict[label_idx] = person_name  # Map label to person name
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize the image data
                    data.append(img)
                    labels.append(label_idx)
            label_idx += 1

    data = np.array(data)
    labels = np.array(labels)
    return data, labels, label_dict

# Load the data
data, labels, label_dict = load_data()

# Step 2: Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 3: Load a pre-trained ResNet model (without the top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Step 4: Add custom layers on top for face classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(label_dict), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Step 5: Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
checkpoint = ModelCheckpoint('face_recognition_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[checkpoint])

# Save the label dictionary for future predictions
np.save('label_dict.npy', label_dict)

print("Fine-tuning complete and model saved.")
