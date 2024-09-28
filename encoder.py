from src.model.util import make_encoder, train_dataset
import tensorflow as tf
import matplotlib.pyplot as plt


def binclassparser_(images, labels):
    bin_labels = tf.where(labels < 5, 0.0, 1.0)
    return images, bin_labels


encoder_bin_train_dataset = train_dataset.map(binclassparser_).batch(32)

if __name__ == "__main__":
    # for images, labels in encoder_bin_train_dataset.take(1):
    #     for image, label in zip(images, labels):
    #         plt.imshow(image.numpy().reshape(28, 28), cmap="gray")
    #         plt.title(label.numpy())
    #         plt.show()
    #         plt.close()
    latent_dim = 1
    encoder = make_encoder(latent_dim)
    encoder.summary()

    encoder.compile(
        optimizer="sgd",
        loss="mae",
        metrics=["mse"],
    )
    encoder.fit(encoder_bin_train_dataset, epochs=100)
