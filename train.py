import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# checks to see if tensorflow is installed
#print("TensorFlow version:", tf.__version__)
#print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# ---------------------------------------------------------------------------

# Create the training and testing datasets
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    'train_data/RGB_224x224/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    'train_data/RGB_224x224/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Create the model
Plant_classifer_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

Plant_classifer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = Plant_classifer_model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# Evaluate on test data
predictions = Plant_classifer_model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save the model
Plant_classifer_model.save('plant_classifier_model.h5')
