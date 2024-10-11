import tensorflow as tf
import math
import numpy as np


@tf.keras.utils.register_keras_serializable()
def ContrastivePositiveMetric(y_true, y_pred):
    # y_true = 0 if the pair is similar, 1 if the pair is dissimilar
    similar_margin = np.cos(math.radians(5))
    sim_pen = tf.where(y_pred > similar_margin, 1.0, 0.0)
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_sum((1.0 - y_true) * sim_pen) / tf.reduce_sum((1.0 - y_true))


x = ContrastivePositiveMetric(
    tf.constant([0, 1, 0, 1]), tf.constant([1.0, 0.2, 1.0, 1.0])
)  # <1>

print(x)
