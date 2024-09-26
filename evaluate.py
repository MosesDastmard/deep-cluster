import tensorflow as tf
from model import test_dataset, encoder_dataset
import pandas as pd
import matplotlib.pyplot as plt


def norm(row):
    values = [row[col] for col in row.keys() if "latent" in col]
    row["norm"] = tf.norm(values).numpy()
    return row


# Load the model from the specified file path
model_path = "encoder.keras"  # Replace with the actual path to your model file
model = tf.keras.models.load_model(model_path)

# Display the model summary
model.summary()
latents = []
for images, labels in test_dataset.batch(256):
    latent = model.predict(images)
    latent_df = pd.DataFrame(
        latent, columns=[f"latent_{i}" for i in range(latent.shape[1])]
    )
    latent_df["label"] = labels.numpy()
    latents.append(latent_df)

# Concatenate the latent representations
latents_df = pd.concat(latents, ignore_index=True)
latents_df = latents_df.apply(norm, axis=1)
print(latents_df["norm"].mean())
latents_df.to_parquet("latent_test.parquet", index=False)


# latents = []
# for images, labels in encoder_dataset.batch(256):
#     latent = model.predict(images)
#     latent_df = pd.DataFrame(
#         latent, columns=[f"latent_{i}" for i in range(latent.shape[1])]
#     )
#     latent_df["label"] = labels.numpy()
#     latents.append(latent_df)

# # Concatenate the latent representations
# latents_df = pd.concat(latents, ignore_index=True)
# latents_df.to_parquet("latent_train.parquet", index=False)
