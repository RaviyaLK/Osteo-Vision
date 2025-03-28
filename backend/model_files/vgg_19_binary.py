import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import ResNet50, EfficientNetB3
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

# Suppress Warnings
warnings.filterwarnings(action="ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data Paths
healthy_dirs = [r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/backend/DatasetUltra/Normal']
osteoporosis_dirs = [r'C:/Users/Metropolitan/OneDrive/Desktop/Osteo-Vision/backend/DatasetUltra/Osteoporosis']

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

# Display class distribution
print("Class distribution:")
print(knee_osteoporosis_df['labels'].value_counts())

# Train-Test Split (stratified)
train_val_set, test_set = train_test_split(knee_osteoporosis_df, test_size=0.2, stratify=knee_osteoporosis_df['labels'], random_state=42)
train_set, val_set = train_test_split(train_val_set, test_size=0.2, stratify=train_val_set['labels'], random_state=42)

print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

# Define preprocessing function
def preprocess_image(image):
    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # Merge back
        updated_lab = cv2.merge((cl, a, b))
        image = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)
    except Exception as e:
        # If error occurs in CLAHE processing, use original image
        print(f"CLAHE processing error: {e}")

    # Apply EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

# Enhanced Data Augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Less augmentation for validation to keep it closer to test conditions
val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_image)

# Set up generators
batch_size = 8  # Smaller batch size for better generalization

train = train_gen.flow_from_dataframe(
    dataframe=train_set,
    x_col="filepaths", y_col="labels",
    target_size=(224, 224),
    color_mode='rgb', class_mode="categorical",
    batch_size=batch_size, shuffle=True
)

val = val_gen.flow_from_dataframe(
    dataframe=val_set,
    x_col="filepaths", y_col="labels",
    target_size=(224, 224),
    color_mode='rgb', class_mode="categorical",
    batch_size=batch_size, shuffle=False
)

test = test_gen.flow_from_dataframe(
    dataframe=test_set,
    x_col="filepaths", y_col="labels",
    target_size=(224, 224),
    color_mode='rgb', class_mode="categorical",
    batch_size=batch_size, shuffle=False
)

# Class weights to address class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_set['labels']).tolist(),  # Convert to list
    y=train_set['labels'].values  # Use numpy array
)

# Make sure to convert to native Python float
class_weights_dict = {i: float(weight) for i, weight in enumerate(class_weights)}
print("Class weights:", class_weights_dict)
# Model Building - Using EfficientNetB3 (better performance than VGG19)
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create custom model with more sophisticated architecture
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs)

# Add custom layers - using residual connections for better gradient flow
x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# First residual block
res = x
x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Add()([x, res])  # Add residual connection
x = Activation('relu')(x)

# Global pooling
x = GlobalAveragePooling2D()(x)

# Dense layers with dropout for regularization
x = Dense(512, activation=None, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(256, activation=None, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.4)(x)

# Output layer
outputs = Dense(2, activation='softmax')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

# Freeze base model layers to prevent overfitting
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=CategoricalCrossentropy(label_smoothing=0.1),  # Label smoothing helps generalization
    metrics=['accuracy']  # Simplified metrics to avoid serialization issues
)

# Model summary
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    'best_osteoporosis_model.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=True  # Add this line
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Two-phase training approach
# Phase 1: Train only the top layers
try:
    history1 = model.fit(
        train,
        epochs=20,
        validation_data=val,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
except Exception as e:
    print(f"Error during phase 1 training: {e}")
    import traceback
    traceback.print_exc()
    # Try to continue anyway
    print("Attempting to continue with phase 2...")

# Phase 2: Fine-tune the model by unfreezing some layers
print("Fine-tuning model...")
# Unfreeze some layers of the base model
for layer in base_model.layers[-30:]:  # Unfreeze the last 30 layers
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']  # Simplified metrics to avoid serialization issues
)

# Continue training
try:
    history2 = model.fit(
        train,
        epochs=30,
        validation_data=val,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
except Exception as e:
    print(f"Error during phase 2 training: {e}")
    import traceback
    traceback.print_exc()

# Try to load the best model
try:
    model.load_weights('best_osteoporosis_model.h5')
    print("Successfully loaded best model weights")
except Exception as e:
    print(f"Error loading best model: {e}")

# Save the final model
try:
    model.save("improved_osteoporosis_model.h5")
    print("Successfully saved final model")
except Exception as e:
    print(f"Error saving final model: {e}")
    # Try alternative save format
    try:
        model.save("improved_osteoporosis_model", save_format="tf")
        print("Successfully saved model in TensorFlow format")
    except Exception as e2:
        print(f"Error saving in TensorFlow format: {e2}")

# Evaluate on test set
try:
    test.reset()
    test_results = model.evaluate(test, verbose=1)
    test_loss, test_accuracy = test_results[0], test_results[1]
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

# Generate predictions
try:
    test.reset()
    predictions = model.predict(test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test.classes
    class_labels = list(test.class_indices.keys())

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Classification Report
    class_report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print("Classification Report:\n", class_report)
except Exception as e:
    print(f"Error generating predictions: {e}")

# Optional: Plot training history if available
def plot_history(history1, history2=None):
    try:
        plt.figure(figsize=(15, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history1.history['accuracy'], label='Train Accuracy (Phase 1)')
        plt.plot(history1.history['val_accuracy'], label='Val Accuracy (Phase 1)')

        if history2:
            # Adjust indices for history2 to continue from history1
            start_epoch = len(history1.history['accuracy'])
            epochs2 = range(start_epoch, start_epoch + len(history2.history['accuracy']))

            plt.plot(epochs2, history2.history['accuracy'], label='Train Accuracy (Phase 2)')
            plt.plot(epochs2, history2.history['val_accuracy'], label='Val Accuracy (Phase 2)')

        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history1.history['loss'], label='Train Loss (Phase 1)')
        plt.plot(history1.history['val_loss'], label='Val Loss (Phase 1)')

        if history2:
            plt.plot(epochs2, history2.history['loss'], label='Train Loss (Phase 2)')
            plt.plot(epochs2, history2.history['val_loss'], label='Val Loss (Phase 2)')

        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    except Exception as e:
        print(f"Error plotting history: {e}")

# Try to plot history if available
try:
    if 'history1' in locals():
        if 'history2' in locals():
            plot_history(history1, history2)
        else:
            plot_history(history1)
except Exception as e:
    print(f"Error plotting history: {e}")


# Function to visualize sample predictions
def display_sample_predictions(generator, model, num_samples=5):
    try:
        generator.reset()
        batch = next(generator)
        images, labels = batch
        
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        
        plt.figure(figsize=(20, 4))
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i + 1)
            
            # Convert the image for display
            img = images[i].copy()
            
            # Attempt to reverse preprocessing for better visualization
            # Note: This is an approximation since exact reversal depends on the preprocessing
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            plt.imshow(img)
            
            # Color-code the prediction text based on correctness
            color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
            
            # Display the prediction and ground truth
            plt.title(f"True: {class_labels[true_classes[i]]}\n"
                    f"Pred: {class_labels[predicted_classes[i]]}\n"
                    f"Conf: {predictions[i][predicted_classes[i]]:.2f}", 
                    color=color)
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png')
        plt.show()
    except Exception as e:
        print(f"Error displaying sample predictions: {e}")

# Try to display sample predictions
try:
    display_sample_predictions(test, model, num_samples=5)
except Exception as e:
    print(f"Error in sample predictions: {e}")

# Create a simple prediction function for new images
def predict_image(model, image_path):
    """
    Make a prediction on a single image
    
    Args:
        model: Trained model
        image_path: Path to the image file
    
    Returns:
        Prediction class and confidence
    """
    try:
        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not load image", 0
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        
        return class_labels[predicted_class], float(confidence)
    except Exception as e:
        return f"Error: {str(e)}", 0

# Example usage for prediction function
print("\nExample prediction:")
try:
    if len(test_set) > 0:
        test_image = test_set['filepaths'].iloc[0]
        true_label = test_set['labels'].iloc[0]
        pred_label, confidence = predict_image(model, test_image)
        print(f"Image: {test_image}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {pred_label} with confidence: {confidence:.2f}")
except Exception as e:
    print(f"Error in example prediction: {e}")
