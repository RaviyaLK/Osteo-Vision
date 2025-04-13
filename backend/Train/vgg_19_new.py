# Import Libraries
import os
import numpy as np
import pandas as pd
import warnings
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings
warnings.filterwarnings("ignore")


# -----------------------------------
# 📌 1. Load Dataset
# -----------------------------------
def load_dataset():
    """
    Loads image file paths and labels into a DataFrame.
    """
    healthy_dirs = [
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Normal/normal',
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Normal/normal-1',
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Normal/normal-2',
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Normal/normal-3'
    ]

    osteoporosis_dirs = [
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Osteoporosis/osteoporosis',
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Osteoporosis/osteoporosis-1',
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Osteoporosis/osteoporosis-2',
        r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/Dataset/Osteoporosis/osteoporosis-3'
    ]

    filepaths, labels = [], []
    class_dirs = [healthy_dirs, osteoporosis_dirs]
    class_labels = ['Healthy', 'Osteoporosis']

    for i, dirs in enumerate(class_dirs):
        for dir_path in dirs:
            for file in os.listdir(dir_path):
                filepaths.append(os.path.join(dir_path, file))
                labels.append(class_labels[i])

    return pd.DataFrame({"filepaths": filepaths, "labels": labels})


# -----------------------------------
# 📌 2. Preprocess Data
# -----------------------------------
def preprocess_data(df):
    """
    Splits dataset into train, validation, and test sets.
    """
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


# -----------------------------------
# 📌 3. Data Generators
# -----------------------------------
def create_generators(train_df, val_df, test_df):
    """
    Creates image generators for training, validation, and testing.
    """
    image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)

    train_gen = image_gen.flow_from_dataframe(
        train_df, x_col="filepaths", y_col="labels", target_size=(224, 224),
        color_mode="rgb", class_mode="categorical", batch_size=16, shuffle=True)

    val_gen = image_gen.flow_from_dataframe(
        val_df, x_col="filepaths", y_col="labels", target_size=(224, 224),
        color_mode="rgb", class_mode="categorical", batch_size=16, shuffle=False)

    test_gen = image_gen.flow_from_dataframe(
        test_df, x_col="filepaths", y_col="labels", target_size=(224, 224),
        color_mode="rgb", class_mode="categorical", batch_size=16, shuffle=False)

    return train_gen, val_gen, test_gen


# -----------------------------------
# 📌 4. Build Model (VGG19 with Custom Layers)
# -----------------------------------
def build_model():
    """
    Builds and returns a fine-tuned VGG19 model.
    """
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False  # Freeze base model layers

    x = base_model.output
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)
    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# -----------------------------------
# 📌 5. Train Model
# -----------------------------------
def train_model(model, train_gen, val_gen):
    """
    Trains the model with early stopping and learning rate scheduler.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    annealer = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

    history = model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=[early_stopping, annealer])
    model.save("knee_osteoporosis_model_V2_new_new.h5")

    return history


# -----------------------------------
# 📌 6. Evaluate Model
# -----------------------------------
def evaluate_model(model, test_gen):
    """
    Evaluates model and prints confusion matrix & classification report.
    """
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss}\nTest Accuracy: {test_acc}")

    test_gen.reset()
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))


# -----------------------------------
# 📌 7. Run Pipeline
# -----------------------------------
if __name__ == "__main__":
    df = load_dataset()
    train_df, val_df, test_df = preprocess_data(df)
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)
    model = build_model()
    history = train_model(model, train_gen, val_gen)
    evaluate_model(model, test_gen)
