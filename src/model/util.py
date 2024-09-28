import tensorflow as tf
from tensorflow.keras import layers, models
import math
import numpy as np
import os


@tf.keras.utils.register_keras_serializable()
class UniqLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.clip_by_value(inputs, -1, 1)


@tf.keras.utils.register_keras_serializable()
class L2NormLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)


class ReluLayer(tf.keras.layers.Layer):
    def __init__(self, break_point=0, max_point=1, neg_slope=0.0, pos_slope=0.0):
        super(ReluLayer, self).__init__()
        assert max_point > break_point, "max_point must be greater than break_point"
        self.break_point = break_point
        self.max_point = max_point
        self.neg_slope = neg_slope
        self.pos_slope = pos_slope

    def call(self, inputs):
        coeff = self.max_point / (self.max_point - self.break_point)
        y = tf.where(
            inputs < self.break_point,
            (inputs - self.break_point) * self.neg_slope,
            tf.where(
                inputs > self.max_point,
                (inputs - self.max_point) * self.pos_slope,
                (inputs - self.break_point) * coeff,
            ),
        )
        return y


class SimilarityLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(SimilarityLayer, self).__init__(**kwargs)

    def Sigmoid(x):
        return 1 / (1 + tf.exp(-((x - 0.3) * 6)))

    def call(self, inputs):
        # Split inputs into two tensors
        vector_a, vector_b = inputs
        # Compute the L2 norms (magnitudes) of each vector

        norm_a = tf.norm(vector_a, axis=-1)
        norm_b = tf.norm(vector_b, axis=-1)
        dot_product = tf.reduce_sum(tf.multiply(vector_a, vector_b), axis=-1) / (
            norm_a * norm_b + 1e-10
        )

        similarity = SimilarityLayer.Sigmoid(dot_product)
        return similarity


if __name__ == "__main__":
    # Example: Cosine of 45 degrees (converted to radians)
    encoder = make_encoder(latent_dim)
    # discriminator = make_discriminator(latent_dim)
    model = make_model(encoder)
    model.summary()

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=1e-3,
        momentum=0.95,  # Momentum helps in smoothing out the updates
        nesterov=True,  # Nesterov momentum is often used in deep learning for better convergence)
    )
    model.compile(
        optimizer=optimizer,
        loss={
            "dense": "binary_crossentropy",
            "similarity": "binary_crossentropy",
        },
        loss_weights={
            "dense": 1.0,
            "similarity": 1.0,
        },
        metrics={
            "dense": ["accuracy"],  # Add accuracy for the 'dense' output
            "similarity": ["mae"],  # Add mae for the 'similarity' output
        },
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True,
        mode="min",
    )
    model.fit(
        model_train_dataset,
        epochs=50,
        validation_data=model_test_dataset,
        callbacks=[early_stopping],
    )

    # onel = models.Sequential(name="onel")
    # onel.add(encoder)
    # onel.add(discriminator)

    encoder.save("encoder.keras")
    # discriminator.save("discriminator.keras")
    model.save("model.keras")
    # onel.save("onel.keras")
