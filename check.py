import tensorflow as tf
from model import encoder_dataset, L2NormLayer
import pandas as pd
import matplotlib.pyplot as plt


def norm(row):
    values = [row[col] for col in row.keys() if "latent" in col]
    row["norm"] = tf.norm(values).numpy()
    return row


# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
model = tf.keras.models.load_model(model_path)

latents = model.predict(encoder_dataset.batch(256))

# compute mean of norm of latent vectors
norms = tf.abs(tf.norm(latents, axis=1).numpy() - 1)
print(tf.reduce_mean(norms))


plt.plot(
    latents[:, 0],
    latents[:, 1],
)
plt.show()
