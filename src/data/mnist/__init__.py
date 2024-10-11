from tensorflow.keras.datasets import mnist
import tensorflow as tf
from ..util import SiameseGenerator

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


train_size = len(train_images)
test_size = len(test_images)


def get_encoder_dataset(batch_size):
    return (
        SiameseGenerator(train_images, train_labels, batch_size=batch_size),
        SiameseGenerator(test_images, test_labels, batch_size=batch_size),
        INPUT_SHAPE,
    )
