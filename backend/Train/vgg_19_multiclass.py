# ===============================
# 1. Imports and Configuration
# ===============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================
# 2. Dataset Paths
# ===============================
healthy_dirs = [
   r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/DatasetUltra/Normal'
]

osteopenia_dirs = [
    r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/DatasetUltra/Osteopenia',
    r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\DatasetUltra\New"
]

osteoporosis_dirs = [
   r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/DatasetUltra/Osteoporosis'
]

# ===============================
# 3. Load Image Paths and Labels
# ===============================
filepaths = []
labels = []
class_dirs = [healthy_dirs, osteopenia_dirs, osteoporosis_dirs]
class_names = ['Healthy', 'Osteopenia', 'Osteoporosis']

for i, dir_list in enumerate(class_dirs):
    for dir_path in dir_list:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    filepaths.append(os.path.join(dir_path, file))
                    labels.append(class_names[i])

# Create DataFrame
df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
print(df['labels'].value_counts())
print(df.head())

# ===============================
# 4. Train-Test Split
# ===============================
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['labels'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['labels'], random_state=42)

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# ===============================
# 5. Image Data Generators
# ===============================
image_size = (224, 224)
batch_size = 4

datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

train_gen = datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                        color_mode='rgb',
                                        target_size=image_size,
                                        class_mode='categorical',
                                        batch_size=batch_size,
                                        shuffle=False)

val_gen = datagen.flow_from_dataframe(val_df, x_col='filepaths', y_col='labels',
                                      color_mode='rgb',
                                      target_size=image_size,
                                      class_mode='categorical',
                                      batch_size=batch_size,
                                      shuffle=False)

test_gen = datagen.flow_from_dataframe(test_df, x_col='filepaths',
                                       y_col='labels', color_mode='rgb',
                                       target_size=image_size,
                                       class_mode='categorical',
                                       batch_size=batch_size,
                                       shuffle=False)

# ===============================
# 6. Model Architecture
# ===============================
model = Sequential([
    Conv2D(128, (8, 8), strides=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),

    Conv2D(256, (5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (1, 1), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (1, 1), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),

    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===============================
# 7. Learning Rate Scheduler
# ===============================
# lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * 0.95 ** epoch)

# ===============================
# 8. Training the Model
# ===============================
history = model.fit(train_gen, validation_data=val_gen, epochs=20,

                    verbose=1)

# ===============================
# 9. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ===============================
# 10. Plot Accuracy Graph
# ===============================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# ===============================
# 11. Save the Model
# ===============================
model.save('Multiclass_Osteoporosis_Model.h5')
print("Model saved as Multiclass_Osteoporosis_Model.h5")
# ===============================
# 12. Classification Report & Confusion Matrix
# ===============================
# Get true labels
true_labels = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# Predict probabilities
y_pred_prob = model.predict(test_gen, verbose=1)

# Convert probabilities to predicted class indices
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(true_labels, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(true_labels, y_pred)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
