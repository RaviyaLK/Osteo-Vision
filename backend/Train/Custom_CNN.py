import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Folder paths
train_dir = r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\Data\Train"
val_dir = r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\Data\Validation"
test_dir = r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\Data\Test"

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Use 'categorical' for multiclass
    shuffle=True
)

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Use 'categorical' for multiclass
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Use 'categorical' for multiclass
    shuffle=False
)
def create_binary_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary output
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    return model
def create_multiclass_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes for multiclass output
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use categorical for multiclass
                  metrics=['accuracy', 'Precision', 'Recall'])

    return model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
binary_model = create_binary_model()

binary_history = binary_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping, model_checkpoint]
)
multiclass_model = create_multiclass_model()

multiclass_history = multiclass_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping, model_checkpoint]
)
binary_model.save('binary_model.h5')  # Save the binary model
multiclass_model.save('multiclass_model.h5')  # Save the multiclass model
binary_model.load_weights('best_model.h5')  # Load the best weights saved by ModelCheckpoint

# Evaluate on the test data
test_loss, test_acc, test_precision, test_recall = binary_model.evaluate(test_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

# Confusion Matrix
test_predictions = binary_model.predict(test_generator)
test_predictions = (test_predictions > 0.5).astype(int)

cm = confusion_matrix(test_generator.classes, test_predictions)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(test_generator.classes, test_predictions))
multiclass_model.load_weights('best_model.h5')  # Load the best weights saved by ModelCheckpoint

# Evaluate on the test data
test_loss, test_acc, test_precision, test_recall = multiclass_model.evaluate(test_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

# Confusion Matrix
test_predictions = multiclass_model.predict(test_generator)
test_predictions = np.argmax(test_predictions, axis=1)

cm = confusion_matrix(test_generator.classes, test_predictions)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(test_generator.classes, test_predictions))
# For Binary Model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(binary_history.history['accuracy'], label='Training Accuracy')
plt.plot(binary_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Binary Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(binary_history.history['loss'], label='Training Loss')
plt.plot(binary_history.history['val_loss'], label='Validation Loss')
plt.title('Binary Model Loss')
plt.legend()
plt.show()

# For Multiclass Model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(multiclass_history.history['accuracy'], label='Training Accuracy')
plt.plot(multiclass_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Multiclass Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(multiclass_history.history['loss'], label='Training Loss')
plt.plot(multiclass_history.history['val_loss'], label='Validation Loss')
plt.title('Multiclass Model Loss')
plt.legend()
plt.show()
