import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.utils import class_weight
import os
import numpy as np  # Import NumPy

# Constants
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE  # For performance optimization

# Function to load and preprocess images with enhanced augmentation
def process_image(file_path, label):
    # Load the image from file
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode the image as RGB
    img = tf.image.resize(img, IMAGE_SIZE)  # Resize the image to the desired size
    img = img / 255.0  # Normalize the pixel values
    
    # Apply random augmentations
    img = tf.image.random_flip_left_right(img)  # Horizontal flip
    img = tf.image.random_flip_up_down(img)  # Vertical flip
    img = tf.image.random_brightness(img, max_delta=0.1)  # Random brightness adjustment
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)  # Random contrast adjustment
    
    # Randomly rotate the image
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # Random rotation (0-3 times 90 degrees)

    # Randomly zoom in and out
    img = tf.image.resize_with_crop_or_pad(img, target_height=int(IMAGE_SIZE[0] * 1.2), target_width=int(IMAGE_SIZE[1] * 1.2))  # Zoom in
    img = tf.image.resize(img, IMAGE_SIZE)  # Resize back to original size

    # Random shearing (simulated with resizing and cropping)
    if tf.random.uniform(()) > 0.5:  # 50% chance to apply shearing
        # Create a shear transformation effect
        shear_factor = 0.2
        img = tf.image.resize(img, [int(IMAGE_SIZE[0] * (1 + shear_factor)), IMAGE_SIZE[1]])  # Resize height
        img = tf.image.resize(img, IMAGE_SIZE)  # Resize back to original size

    return img, label

# Function to load data from directories and create a dataset
def load_dataset(data_dir, batch_size, shuffle=True):
    classes = ['thief', 'non_thief']  # Ensure these match your folder names
    
    # Get the list of file paths and labels
    file_paths = []
    labels = []
    for idx, label in enumerate(classes):
        class_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, file_name))
            labels.append(idx)  # 0 for thief, 1 for non_thief
    
    # Create TensorFlow Dataset object
    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Map the dataset to load and preprocess the images
    dataset = dataset.map(process_image, num_parallel_calls=AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths))  # Shuffle the dataset
    
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)  # Batch and prefetch for performance
    return dataset

# Build the model
def build_model():
    model = Sequential()

    model.add(Input(shape=(150, 150, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to avoid overfitting
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(train_dataset, validation_dataset):
    model = build_model()

    # Handle class imbalance
    y_labels = []
    for _, label in train_dataset.unbatch():
        y_labels.append(label.numpy())
    
    # Convert the list of labels to a NumPy array
    y_labels = np.array(y_labels)

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),  # Convert to NumPy array
        y=y_labels
    )
    class_weights = dict(enumerate(class_weights))

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,  # Adjust the number of epochs as needed
        class_weight=class_weights  # Pass class weights to handle imbalance
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(validation_dataset)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

    # Save the model after training
    model.save('my_model.h5')
    return model

if __name__ == "__main__":
    # Paths to the train and validation directories
    train_dir = 'dataset/train'  # Your train data directory
    validation_dir = 'dataset/validation'  # Your validation data directory

    # Load the datasets
    train_dataset = load_dataset(train_dir, batch_size=BATCH_SIZE)
    validation_dataset = load_dataset(validation_dir, batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    model = train_model(train_dataset, validation_dataset)
    model.compile(optimizer='adam', loss='your_loss_function', metrics=['accuracy'])

