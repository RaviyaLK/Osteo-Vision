# Import Libraries
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
# Suppress Warnings
warnings.filterwarnings("ignore")

# Learning Rate Scheduler
annealer = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch, verbose=0)

# Data Paths
healthy_dirs = [r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/DatasetUltra/Normal']
osteoporosis_dirs = [r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/DatasetUltra/Osteoporosis']
# Prepare Dataset
filepaths = []
labels = []
dict_lists = [healthy_dirs, osteoporosis_dirs]
class_labels = ['Healthy', 'Osteoporosis']

for i, dir_list in enumerate(dict_lists):
    for j in dir_list:
        flist = os.listdir(j)
        for f in flist:
            fpath = os.path.join(j, f)
            filepaths.append(fpath)
            labels.append(class_labels[i])

# Create DataFrame
Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
knee_osteoporosis_data = pd.concat([Fseries, Lseries], axis=1)
knee_osteoporosis_df = pd.DataFrame(knee_osteoporosis_data)

# Train-Test Split
train_images, test_images = train_test_split(knee_osteoporosis_df, test_size=0.3, random_state=42)
train_set, val_set = train_test_split(knee_osteoporosis_df, test_size=0.2, random_state=42)

# Image Data Generators
image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)
train = image_gen.flow_from_dataframe(
    dataframe=train_set,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=16,
    shuffle=True
)
test = image_gen.flow_from_dataframe(
    dataframe=test_images,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=16,
    shuffle=False
)
val = image_gen.flow_from_dataframe(
    dataframe=val_set,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=16,
    shuffle=False
)

# Model Setup and Fine-Tuning
vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = vgg19_model.output

# Custom Layers for Classification
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation="relu")(x)
output = Dense(2, activation="softmax")(x)

# Final Model
model = Model(inputs=vgg19_model.input, outputs=output)

# Freeze Pre-trained Layers
for layer in vgg19_model.layers:
    layer.trainable = False

# Compile Model
model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model Training with Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
annealer = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)
history = model.fit(
    train,
    epochs=20,
    validation_data=val,
    callbacks=[early_stopping, annealer]
)

# Save Model
model.save("knee_osteoporosis_model_Anthima_eka.h5")

# Model Evaluation
test_loss, test_accuracy = model.evaluate(test, verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate Predictions for Confusion Matrix and Classification Report
# Reset the test generator before making predictions
test.reset()

# Predict the classes for the test set
predictions = model.predict(test, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test.classes
class_labels = list(test.class_indices.keys())

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for VGG19 Model on Test Set')
plt.show()

# Classification Report
class_report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:\n", class_report)

# Optional: Display Sample Predictions
def display_sample_predictions(generator, model, num_samples=5):
    import matplotlib.pyplot as plt
    import cv2

    # Get a batch of images and labels
    images, labels = next(generator)
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        img = images[i].astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.imshow(img)
        plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[predicted_classes[i]]}")
        plt.axis('off')
    plt.show()

# Display Sample Predictions
display_sample_predictions(test, model, num_samples=5)
