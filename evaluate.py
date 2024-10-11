import tensorflow as tf
from src.data.mnist import test_dataset as mnist_test_dataset
from src.data.fashion import test_dataset as fashion_test_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
Decomposer = PCA
scaler = StandardScaler()
decomposer = Decomposer(n_components=2)


encoder = tf.keras.models.load_model(model_path)


latents = []
for images, labels in mnist_test_dataset.batch(1024):
    latent = encoder.predict(images)
    latent_df = pd.DataFrame(
        latent, columns=[f"latent_{i}" for i in range(latent.shape[1])]
    )
    latent_df["label"] = labels.numpy()
    latents.append(latent_df)


# Concatenate the latent representations
latents_df = pd.concat(latents, ignore_index=True)
latents_df.to_parquet("latents.parquet")
latents = latents_df.drop("label", axis=1)


latents_scaled = scaler.fit_transform(latents)
# Perform PCA on the latent representations

components = decomposer.fit_transform(latents_scaled)


# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(
    components[:, 0],
    components[:, 1],
    c=latents_df["label"],
    cmap="tab10",
    s=200,
)
plt.show()
latents = []
for images, labels in fashion_test_dataset.batch(1024):
    latent = encoder.predict(images)
    latent_df = pd.DataFrame(
        latent, columns=[f"latent_{i}" for i in range(latent.shape[1])]
    )
    latent_df["label"] = -1
    latents.append(latent_df)

# Concatenate the latent representations
latents_df = pd.concat(latents, ignore_index=True)
latents = latents_df.drop("label", axis=1)


latents_scaled = scaler.transform(latents)
# Perform PCA on the latent representations

components = decomposer.transform(latents_scaled)
plt.scatter(
    components[:, 0],
    components[:, 1],
    s=200,
)
plt.show()
