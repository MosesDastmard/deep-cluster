from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from model import data_augmentation
import tensorflow as tf

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


aug_imgage = data_augmentation(tf.expand_dims(train_images[10], axis=-1))
plt.imshow(aug_imgage)
plt.show()
