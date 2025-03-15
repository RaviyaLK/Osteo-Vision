#aluth eka
# Import Libraries
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Suppress Warnings
warnings.filterwarnings(action="ignore")

# Enhanced Image Preprocessing Function
def preprocess_image(image):
    # Convert to float32
    image = tf.cast(image, tf.float32)
    # Apply CLAHE-like enhancement
    image = tf.image.adjust_contrast(image, 1.3)
    image = tf.image.adjust_brightness(image, 0.1)
    # Apply EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

# Create Model Function
def create_model(input_shape=(380, 380, 3)):
    # Use EfficientNetB4 with noisy student weights for better initialization
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    # Enhanced Dense layers with L2 regularization
    x = Dense(1536, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(768, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(384, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model

# Enhanced Image Data Generators with Stronger Augmentation
train_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='reflect',
    preprocessing_function=preprocess_image
)

valid_gen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)


# Data Paths
healthy_dirs = [
    r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/DatasetUltra/Normal'

]
osteoporosis_dirs = [
    r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/DatasetUltra/Osteoporosis'
]


# Prepare Dataset
filepaths = []
labels = []
for i, dir_list in enumerate([healthy_dirs, osteoporosis_dirs]):
    for directory in dir_list:
        for f in os.listdir(directory):
            filepaths.append(os.path.join(directory, f))
            labels.append(['Healthy', 'Osteoporosis'][i])

# Create DataFrame
df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Stratified K-Fold Cross-validation
n_splits = 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_val_accuracy = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepaths'], df['labels'])):
    print(f'\nTraining Fold {fold + 1}/{n_splits}')

    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]

    # Further split validation data into validation and test
    val_data, test_data = train_test_split(
        val_data,
        test_size=0.5,
        stratify=val_data['labels'],
        random_state=42
    )

    # Initialize generators
    train = train_gen.flow_from_dataframe(
        dataframe=train_data,
        x_col="filepaths",
        y_col="labels",
        target_size=(380, 380),
        batch_size=12,
        class_mode="categorical"
    )

    val = valid_gen.flow_from_dataframe(
        dataframe=val_data,
        x_col="filepaths",
        y_col="labels",
        target_size=(380, 380),
        batch_size=12,
        class_mode="categorical"
    )

    # Create and compile model
    model, base_model = create_model()

    # First phase: Train only top layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint(
        f'best_model_fold_{fold}.h5',  # Changed from .h5 to .keras
        monitor='val_accuracy',
        save_best_only=True
    )
    ]
    # Initial training
    model.fit(
        train,
        epochs=10,
        validation_data=val,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 1.2}  # Slight class weighting
    )

    # Fine-tuning phase
    for layer in base_model.layers[-50:]:  # Unfreeze more layers
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tuning training
    history = model.fit(
        train,
        epochs=20,
        validation_data=val,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 1.2}
    )

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model
        model.save("knee_osteoporosis_model_efficientNet.h5")

# Final evaluation on test set
test = valid_gen.flow_from_dataframe(
    dataframe=test_data,
    x_col="filepaths",
    y_col="labels",
    target_size=(380, 380),
    batch_size=12,
    class_mode="categorical",
    shuffle=False
)

test_loss, test_accuracy = best_model.evaluate(test)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

# Generate predictions and metrics
predictions = best_model.predict(test)
y_pred = np.argmax(predictions, axis=1)
y_true = test.classes

# Display confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=test.class_indices.keys())
cm_display.plot(cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test.class_indices.keys()))
