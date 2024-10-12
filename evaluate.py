import tensorflow as tf
from src.data.mnist import (
    test_images as mnist_test_images,
    test_labels as mnist_test_labels,
)


from src.data.fashion import (
    test_images as fashion_test_images,
    test_labels as fashion_test_labels,
)

# from src.data.fashion import test_dataset as fashion_test_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
Decomposer = TSNE
scaler = StandardScaler()
decomposer = Decomposer(n_components=2)


encoder = tf.keras.models.load_model(model_path)


latent = encoder.predict(mnist_test_images)
latent_df = pd.DataFrame(
    latent, columns=[f"latent_{i}" for i in range(latent.shape[1])]
)
latent_df["label"] = mnist_test_labels

latent = encoder.predict(fashion_test_images)
fashion_df = pd.DataFrame(
    latent,
    columns=[f"latent_{i}" for i in range(latent.shape[1])],
)
fashion_df["label"] = 10

latent_df = pd.concat([latent_df, fashion_df], ignore_index=True)
latents = latent_df.drop("label", axis=1)


latents_scaled = scaler.fit_transform(latents)
# Perform PCA on the latent representations

components = decomposer.fit_transform(latents_scaled)


# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(
    components[:, 0],
    components[:, 1],
    c=latent_df["label"],
    cmap="tab10",
    s=200,
)

filtered_components = components[latent_df["label"] != 10]
filtered_labels = latent_df[latent_df["label"] != 10]

plt.figure(figsize=(8, 8))
plt.scatter(
    filtered_components[:, 0],
    filtered_components[:, 1],
    c=filtered_labels["label"],
    cmap="tab10",
    s=200,
)
plt.show()
