import tensorflow as tf
from src.data.mnist import (
    test_images as mnist_test_dataset,
    test_labels as mnist_test_labels,
)
from src.data.fashion import test_dataset as fashion_test_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
encoder = tf.keras.models.load_model(model_path)


latents = encoder.predict(mnist_test_dataset)
latent_df = pd.DataFrame(
    latents, columns=[f"latent_{i}" for i in range(latents.shape[1])]
)
latent_df["label"] = mnist_test_labels


# # Concatenate the latent representations
# rows = latent_df.to_dict(orient="records")
# pair = np.random.choice(rows, 2)


# def get_loss(pair):
#     a, b = pair
#     encode_a = [a[f"latent_{i}"] for i in range(2)]
#     encode_b = [b[f"latent_{i}"] for i in range(2)]

#     cosine_distance = -(np.sum([i * j for i, j in zip(encode_a, encode_b)]) - 1)
#     label = (a["label"] != b["label"]) * 1
#     square_pred = np.square(cosine_distance)
#     margin = 1
#     margine_square = np.square(max(0, margin - cosine_distance))
#     loss = label * margine_square + (1 - label) * square_pred
#     return loss


# losses = []
# while True:
#     losses.append(get_loss(np.random.choice(rows, 2)))
#     print(np.mean(losses))
#     if len(losses) > 10000:
#         break
# exit()

# get_loss(pair)

# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(
    latent_df["latent_0"],
    latent_df["latent_1"],
    c=latent_df["label"],
    cmap="tab10",
    s=200,
)
plt.show()
exit()
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
plt.scatter(
    latents_df["latent_0"],
    latents_df["latent_1"],
    s=200,
)
plt.show()
