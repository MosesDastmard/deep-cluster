import tensorflow as tf
from src.model.util import train_dataset, train_dataset
import pandas as pd
import matplotlib.pyplot as plt


def norm(row):
    values = [row[col] for col in row.keys() if "latent" in col]
    row["norm"] = tf.norm(values).numpy()
    return row


# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
model = tf.keras.models.load_model(model_path)

latents = []
for images, labels in train_dataset.batch(2**16):
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    latent = model.predict(images)
    latent_df = pd.DataFrame(
        latent, columns=[f"latent_{i}" for i in range(latent.shape[1])]
    )
    latent_df["label"] = labels.numpy()
    latents.append(latent_df)

# Concatenate the latent representations
latents_df = pd.concat(latents, ignore_index=True)


labels = []
for images, label in train_dataset:
    labels.append(label)
labels = tf.concat(labels, axis=0).numpy()
print("% of positive class: ", sum(labels) / len(labels))


plt.scatter(
    latents_df["latent_0"],
    latents_df["latent_1"],
    c=latents_df["label"],
    cmap="viridis",
)
plt.colorbar()
plt.show()
