import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import math
import numpy as np

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to [0, 1] range
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

train_filter = np.isin(train_labels, [1, 8])
test_filter = np.isin(test_labels, [1, 8])

train_labels = train_labels[train_filter]
train_images = train_images[train_filter]

test_images = test_images[test_filter]
test_labels = test_labels[test_filter]

# Reshape the images to add the channel dimension (batch_size, height, width, channels)
train_images = train_images[
    ..., tf.newaxis
]  # Adding the channel dimension for grayscale
test_images = test_images[..., tf.newaxis]

# Convert training and testing datasets into tf.data.Dataset
encoder_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Define parameters
batch_size = 64 * 2
shuffle_buffer_size = batch_size * 16


def shuffle__images(images, labels):
    # Get the number of elements along the first axis (number of rows)
    num_samples = tf.shape(images)[0]

    # Generate random indices and shuffle them
    indices = tf.random.shuffle(tf.range(num_samples))

    # Apply the same shuffle order to both tensors
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    return shuffled_images, shuffled_labels


def negative_pairs__(images, labels, shuffled_images, shuffled_labels):

    pair_labels = tf.cast((shuffled_labels == labels), tf.float32)

    return images, shuffled_images, pair_labels


def positive_pairs__(images, labels, shuffled_images, shuffled_labels):

    order = tf.argsort(labels)
    images = tf.gather(images, order)
    labels = tf.gather(labels, order)

    order_shuffled = tf.argsort(shuffled_labels)
    shuffled_images = tf.gather(shuffled_images, order_shuffled)
    shuffled_labels = tf.gather(shuffled_labels, order_shuffled)

    pair_labels = tf.cast((shuffled_labels == labels), tf.float32)
    return images, shuffled_images, pair_labels


def pair_augmentation(images, labels):
    shuffled_images, shuffled_labels = shuffle__images(images, labels)
    pos_images, pos_shuffled_images, pos_pair_labels = positive_pairs__(
        images, labels, shuffled_images, shuffled_labels
    )
    neg_images, neg_shuffled_images, neg_pair_labels = negative_pairs__(
        images, labels, shuffled_images, shuffled_labels
    )
    images = tf.concat([pos_images, neg_images], axis=0)
    shuffled_images = tf.concat([pos_shuffled_images, neg_shuffled_images], axis=0)
    pair_labels = tf.concat([pos_pair_labels, neg_pair_labels], axis=0)
    return (images, shuffled_images), pair_labels


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(
            0.15, value_range=(0, 1), fill_mode="constant", fill_value=0
        ),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.05),
        layers.RandomBrightness(
            0.05,
            value_range=(0, 1),
        ),
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05), 0, 1
            )
        ),
    ]
)


# Preprocessing and batching for training dataset
train_dataset = (
    encoder_dataset.shuffle(shuffle_buffer_size)
    .batch(batch_size)  # Batch the dataset
    .repeat(16)
    .map(
        pair_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )  # Apply augmentation
    .prefetch(tf.data.experimental.AUTOTUNE)
)  # Prefetch for performance

# Preprocessing and batching for testing dataset
encoder_test_dataset = (
    test_dataset.batch(batch_size)
    .map(pair_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .prefetch(tf.data.experimental.AUTOTUNE)  # Batch the dataset
)  # Prefetch for performance

# Check a batch
for images, labels in train_dataset.take(1):
    print(f"Batch images shape: {images[0].shape}")
    print(f"Batch labels shape: {labels.shape}")

leaky_relu_layer = layers.LeakyReLU(negative_slope=0.1)


@tf.keras.utils.register_keras_serializable()
class L2NormLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)


# Define the CNN encoder model
def make_encoder(latent_dim):
    model = models.Sequential()

    # Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (5, 5)))
    model.add(leaky_relu_layer)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5)))
    model.add(leaky_relu_layer)
    model.add(layers.Conv2D(128, (3, 3)))
    model.add(leaky_relu_layer)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dim))
    model.add(leaky_relu_layer)
    model.add(layers.Dense(latent_dim))
    model.add(leaky_relu_layer)
    model.add(layers.Dense(latent_dim))
    model.add(leaky_relu_layer)
    model.add(layers.Dense(latent_dim, activation="linear"))
    # model.add(L2NormLayer())
    return model


class ReluLayer(tf.keras.layers.Layer):
    def __init__(self, break_point=0, max_point=1, neg_slope=0.0, pos_slope=0.0):
        super(ReluLayer, self).__init__()
        assert max_point > break_point, "max_point must be greater than break_point"
        self.break_point = break_point
        self.max_point = max_point
        self.neg_slope = neg_slope
        self.pos_slope = pos_slope

    def call(self, inputs):
        coeff = self.max_point / (self.max_point - self.break_point)
        y = tf.where(
            inputs < self.break_point,
            (inputs - self.break_point) * self.neg_slope,
            tf.where(
                inputs > self.max_point,
                (inputs - self.max_point) * self.pos_slope,
                (inputs - self.break_point) * coeff,
            ),
        )
        return y


class SimilarityLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(SimilarityLayer, self).__init__(**kwargs)
        self.relu = ReluLayer(break_point=0, pos_slope=0.1, neg_slope=0.1)

    def call(self, inputs):
        # Split inputs into two tensors
        vector_a, vector_b = inputs

        # Compute the dot product of the vectors

        # Compute the L2 norms (magnitudes) of each vector
        norm_a = tf.sqrt(tf.reduce_sum(tf.square(vector_a), axis=-1))
        norm_b = tf.sqrt(tf.reduce_sum(tf.square(vector_b), axis=-1))

        # Compute cosine similarity
        dot_product = tf.reduce_sum(tf.multiply(vector_a, vector_b), axis=-1)
        similarity = dot_product / (norm_a * norm_b + tf.keras.backend.epsilon())

        similarity = self.relu(similarity)

        return similarity


def make_model(encoder):
    input_a = layers.Input(shape=(28, 28, 1))
    input_a_ = data_augmentation(input_a)
    input_b = layers.Input(shape=(28, 28, 1))
    input_b_ = data_augmentation(input_b)

    # Get the encoded representations of the inputs
    encoded_a = encoder(input_a_)
    encoded_b = encoder(input_b_)

    # Compute the cosine similarity between the encoded vectors
    similarity = SimilarityLayer()([encoded_a, encoded_b])

    # Create the model
    model = models.Model(
        inputs=[input_a, input_b],
        outputs=similarity,
    )

    return model


if __name__ == "__main__":
    # Example: Cosine of 45 degrees (converted to radians)
    latent_dim = 64
    encoder = make_encoder(latent_dim)
    encoder.summary()
    model = make_model(encoder)

    # optimizer = tf.keras.optimizers.SGD(
    #     learning_rate=0.1,
    #     clipvalue=1.0,
    #     momentum=0.95,  # Momentum helps in smoothing out the updates
    #     nesterov=True,  # Nesterov momentum is often used in deep learning for better convergence)
    # )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["mae", "mse"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        mode="min",
    )
    model.fit(
        train_dataset,
        epochs=100,
        validation_data=encoder_test_dataset,
        # callbacks=[early_stopping],
    )

    encoder.save("encoder.keras")
    model.save("model.keras")
