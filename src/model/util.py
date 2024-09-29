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
        abs_diff = tf.abs(vector_a - vector_b)
        distance = 1 - tf.reduce_mean(abs_diff, axis=-1, keepdims=True)

        similarity = SimilarityLayer.Sigmoid(distance)
        return similarity
