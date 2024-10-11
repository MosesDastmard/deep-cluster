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


@tf.keras.utils.register_keras_serializable()
def ContrastiveLoss(y_true, y_pred):
    # y_true = 0 if the pair is similar, 1 if the pair is dissimilar
    sim_err = -(y_pred - 1.0) / 2
    dissimilar_margin = np.cos(math.radians(30))
    dis_err = tf.maximum(0.0, -(dissimilar_margin - y_pred))
    return tf.reduce_mean(y_true * dis_err + (1 - y_true) * sim_err)


@tf.keras.utils.register_keras_serializable()
def ContrastivePositiveMetric(y_true, y_pred):
    # y_true = 0 if the pair is similar, 1 if the pair is dissimilar
    # count postive samples
    if y_pred.shape != y_true.shape:
        raise ValueError(
            "y_pred and y_true must have the same shape, got %s and %s"
            % (y_pred.shape, y_true.shape)
        )
    # y_true = 0 if the pair is similar, 1 if the pair is dissimilar
    similar_margin = np.cos(math.radians(5))
    sim_pen = tf.where(y_pred > similar_margin, 1.0, 0.0)
    y_true = tf.cast(y_true, tf.float32)
    # return tf.reduce_sum((1.0 - y_true))
    return tf.reduce_sum((1.0 - y_true) * sim_pen) / tf.reduce_sum((1.0 - y_true))


@tf.keras.utils.register_keras_serializable()
def ContrastiveNegativeMetric(y_true, y_pred):
    if y_pred.shape != y_true.shape:
        raise ValueError(
            "y_pred and y_true must have the same shape, got %s and %s"
            % (y_pred.shape, y_true.shape)
        )
    # y_true = 0 if the pair is similar, 1 if the pair is dissimilar
    dissimilar_margin = np.cos(math.radians(30))
    dis_pen = tf.where(y_pred < dissimilar_margin, 1.0, 0.0)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    # return tf.reduce_sum((y_true))
    return tf.reduce_sum(y_true * dis_pen) / tf.reduce_sum(y_true)
