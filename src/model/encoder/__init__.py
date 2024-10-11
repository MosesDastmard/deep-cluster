from tensorflow.keras import layers, models
import tensorflow as tf
from src.model.util import (
    ContrastiveLoss,
    ContrastiveNegativeMetric,
    ContrastivePositiveMetric,
    L2NormLayer,
    SimilarityLayer,
)
from src.model.discriminator import make_discriminator
from tensorflow.keras.initializers import Constant

relu_layer = layers.ReLU(negative_slope=0.1)


@tf.keras.utils.register_keras_serializable()
class CosineDistance(tf.keras.layers.Layer):
    def call(self, inputs):
        vector_a, vector_b = inputs
        return tf.reduce_sum(tf.multiply(vector_a, vector_b), axis=-1)


@tf.keras.utils.register_keras_serializable()
class EuqDistance(tf.keras.layers.Layer):
    def call(self, inputs):
        vector_a, vector_b = inputs
        return (
            tf.sqrt(
                tf.reduce_sum(tf.square(vector_a - vector_b), axis=1, keepdims=True)
            )
            + 1e-6
        )


# Define the CNN encoder model
def make_encoder(input_shape, latent_dim):
    model = models.Sequential(name="encoder")
    # Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(64, (5, 5), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(8, (3, 3), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    # model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(4 * latent_dim, activation="leaky_relu"))
    model.add(layers.Dense(1 * latent_dim, activation="tanh"))
    model.add(L2NormLayer())
    return model


def make_ecoder_train_model(encoder):
    input_a = layers.Input(shape=encoder.input_shape[1:], name="images_a")
    input_b = layers.Input(shape=encoder.input_shape[1:], name="images_b")

    # Get the encoded representations of the inputs
    encoded_a = encoder(input_a)
    encoded_b = encoder(input_b)

    # Compute the Euclidean distance between the two output vectors
    euclidian_distance = EuqDistance()([encoded_a, encoded_b])

    cosine_distance = CosineDistance()([encoded_a, encoded_b])

    encoder_train_model = models.Model(
        inputs={
            "images_a": input_a,
            "images_b": input_b,
        },
        outputs=cosine_distance,
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=1.0e-6,  # Learning rate
        clipnorm=10.0,  # Clip the gradients by norm
        momentum=0.95,  # Momentum helps in smoothing out the updates
        nesterov=True,  # Nesterov momentum is often used in deep learning for better convergence)
    )
    encoder_train_model.compile(
        optimizer=optimizer,
        loss=ContrastiveLoss,
        metrics=[ContrastivePositiveMetric, ContrastiveNegativeMetric],
    )
    return encoder, encoder_train_model


def make_discriminator_train_model(discriminator):
    return None, None


def get_models(input_shape, latent_dim=64):
    encoder = make_encoder(input_shape, latent_dim)
    encoder, encoder_train_model = make_ecoder_train_model(encoder)
    discriminator = make_discriminator(encoder.output_shape[-1])
    discriminator, discriminator_train_model = make_discriminator_train_model(
        discriminator
    )
    return encoder, encoder_train_model, discriminator, discriminator_train_model
