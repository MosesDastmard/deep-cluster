import tensorflow as tf
from src.data.mnist import (
    test_images as mnist_test_images,
    test_labels as mnist_test_labels,
    train_images as mnist_train_images,
)


from src.data.fashion import (
    test_images as fashion_test_images,
    test_labels as fashion_test_labels,
)

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
encoder = tf.keras.models.load_model(model_path)


latent = encoder.predict(mnist_train_images)

clf = IsolationForest(n_estimators=500, bootstrap=True, random_state=42, max_features=2)
clf.fit(latent)

plt.figure(figsize=(8, 8))
scores = clf.score_samples(latent)
plt.hist(scores, bins=100, label="MNIST_train", alpha=0.5)

latent = encoder.predict(mnist_test_images)

scores = clf.score_samples(latent)
plt.hist(scores, bins=100, label="MNIST_test", alpha=0.5)

latent = encoder.predict(fashion_test_images)

scores = clf.score_samples(latent)
plt.hist(scores, bins=100, label="Fashion", alpha=0.5)
plt.legend()
plt.show()
