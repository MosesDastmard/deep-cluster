from model import ReluLayer
import tensorflow as tf

relu = ReluLayer(min_value=0.5, max_value=2)
print(relu(tf.constant([-1.0, 0.0, 1.0, 2.0, 3.0])))
