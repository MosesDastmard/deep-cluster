import tensorflow as tf
import matplotlib.pyplot as plt


def Sigmoid(x):
    return 1 / (1 + tf.exp(-((x - 0.3) * 6)))


x = tf.range(-1, 1, 0.01)
y = Sigmoid(x)


plt.plot(x, y)
plt.show()
