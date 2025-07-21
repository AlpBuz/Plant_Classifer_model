import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# checks to see if tensorflow is installed
#print("TensorFlow version:", tf.__version__)
#print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# ---------------------------------------------------------------------------



# create the training and testing datasets

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
    class_mode='categorical'
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
Plant_classifer_model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)


# Save the model
Plant_classifer_model.save('plant_classifier_model.h5')