import tensorflow as tf
from src.data.mnist import test_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
model = tf.keras.models.load_model(model_path)


latents = []
for images, labels in test_dataset.batch(1024):
    latent = model.predict(images)
    latent_df = pd.DataFrame(
        latent, columns=[f"latent_{i}" for i in range(latent.shape[1])]
    )
    latent_df["label"] = labels.numpy()
    latents.append(latent_df)

# Concatenate the latent representations
latents_df = pd.concat(latents, ignore_index=True)

latents = latents_df.drop("label", axis=1)


def norm_scale(x):
    vector = x.values
    norm = np.linalg.norm(vector)
    vector = vector / norm
    return pd.Series(vector, index=x.index)


latents = latents.apply(norm_scale, axis=0)
scaler = StandardScaler()
latents_scaled = scaler.fit_transform(latents)
Decomposer = TSNE
# Perform PCA on the latent representations
decomposer = Decomposer(n_components=2)

components = decomposer.fit_transform(latents_scaled)

print(latent_df.shape)

# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(
    components[:, 0],
    components[:, 1],
    c=latents_df["label"],
    cmap="viridis",
    s=200,
)
plt.show()
