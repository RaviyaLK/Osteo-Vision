import os
import numpy as np
import pandas as pd
import warnings
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from models.model_loader import load_model

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

model = load_model()
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
# 📌 3. Image Augmentation & Data Generators
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




def evaluate_model(model, test_gen):
    """
    Evaluates the model and prints confusion matrix, classification report, and ROC curves.
    """
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss}\nTest Accuracy: {test_acc}")

    # Reset test generator
    test_gen.reset()

    # Get true labels and predictions
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.array(test_gen.classes)  # Ensure true_classes is a NumPy array
    class_labels = list(test_gen.class_indices.keys())

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("\nClassification Report:\n", classification_report(true_classes, predicted_classes, target_names=class_labels))

    # ROC Curves
    plt.figure(figsize=(8, 6))
    for i in range(len(class_labels)):  # Iterate through each class
        y_true_binary = np.array(true_classes == i, dtype=int)  # Convert to binary NumPy array
        fpr, tpr, _ = roc_curve(y_true_binary, predictions[:, i])  # Compute ROC curve
        roc_auc = auc(fpr, tpr)  # Compute AUC
        plt.plot(fpr, tpr, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = load_dataset()
    train_df, val_df, test_df = preprocess_data(df)
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)
    evaluate_model(model, test_gen)
