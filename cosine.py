import numpy as np
import tensorflow as tf
import math

x = tf.sqrt(tf.reduce_sum(tf.square([1, 1, 1]), axis=-1))
print(x)
