import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Concatenate the latent representations
latent_df = pd.read_parquet("latent_train.parquet")
latents = latent_df.drop("label", axis=1)
scaler = StandardScaler()
latents_scaled = scaler.fit_transform(latents)

# Perform PCA on the latent representations
pca = PCA(n_components=2)

components = pca.fit_transform(latents_scaled)

# latent_df = latent_df.apply(norm, axis=1)
print(latent_df.shape)

# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(
    components[:, 0],
    components[:, 1],
    c=latent_df["label"],
    cmap="viridis",
    s=200,
)
plt.show()


# Concatenate the latent representations
latent_df = pd.read_parquet("latent_test.parquet")
latents = latent_df.drop("label", axis=1)
scaler = StandardScaler()
latents_scaled = scaler.fit_transform(latents)

# Perform PCA on the latent representations
pca = PCA(n_components=2)

components = pca.fit_transform(latents_scaled)

# latent_df = latent_df.apply(norm, axis=1)
print(latent_df.shape)

# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(
    components[:, 0],
    components[:, 1],
    c=latent_df["label"],
    cmap="viridis",
    s=200,
)
plt.show()
