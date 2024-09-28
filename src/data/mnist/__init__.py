from tensorflow.keras.datasets import mnist
import tensorflow as tf
from ..util import get_encoder_dataset__, get_discriminator_dataset__

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
INPUT_SHAPE = (28, 28, 1)

# Normalize the images to [0, 1] range
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# Reshape the images to add the channel dimension (batch_size, height, width, channels)
train_images = train_images[
    ..., tf.newaxis
]  # Adding the channel dimension for grayscale
test_images = test_images[..., tf.newaxis]

# Convert training and testing datasets into tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))


def get_encoder_dataset(batch_size):
    return get_encoder_dataset__(train_dataset, test_dataset, INPUT_SHAPE, batch_size)


def get_discriminator_dataset(encoder, batch_size):
    encoder.trainable = False
    return get_discriminator_dataset__(
        encoder, train_dataset, test_dataset, INPUT_SHAPE, batch_size
    )
