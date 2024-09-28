from tensorflow.keras import layers, models
import tensorflow as tf


def make_discriminator(latent_dim):
    model = models.Sequential(name="discriminator")
    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def get_model(encoder):
    encoder.trainable = False
    discriminator = make_discriminator(encoder.output_shape[-1])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=1e-3,
        momentum=0.95,  # Momentum helps in smoothing out the updates
        nesterov=True,  # Nesterov momentum is often used in deep learning for better convergence)
    )
    discriminator.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return discriminator
