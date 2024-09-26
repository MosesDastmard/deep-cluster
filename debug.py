import tensorflow as tf
from model import SimilarityLayer, train_dataset, make_encoder, make_model
import pandas as pd
import matplotlib.pyplot as plt

encoded_a = tf.constant([[1.0, 0.0, 0.0]])
encoded_b = tf.constant([[1.0, 0.0, 1.0]])

similarity, norm_a, norm_b = SimilarityLayer()([encoded_a, encoded_b])
print(similarity.numpy(), norm_a.numpy(), norm_b.numpy())
