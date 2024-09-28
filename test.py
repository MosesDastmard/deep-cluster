import tensorflow as tf

# Load the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the data (values are between 0 and 255, so normalize to 0-1 range)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Check the shape of the dataset
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)
