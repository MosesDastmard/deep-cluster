import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import math

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to [0, 1] range
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

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


def negative_pairs(images, labels):
    # Get the number of elements along the first axis (number of rows)
    num_samples = tf.shape(images)[0]

    # Generate random indices and shuffle them
    indices = tf.random.shuffle(tf.range(num_samples))

    # Apply the same shuffle order to both tensors
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    pair_labels = tf.cast((shuffled_labels == labels), tf.float32)

    norm = tf.cast((pair_labels == pair_labels), tf.float32)
    return images, shuffled_images, pair_labels, norm


def positive_augmentation(images, labels):
    # Get the number of elements along the first axis (number of rows)
    num_samples = tf.shape(images)[0]

    # Generate random indices and shuffle them
    indices = tf.random.shuffle(tf.range(num_samples))

    # Apply the same shuffle order to both tensors
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    order = tf.argsort(labels)
    images = tf.gather(images, order)
    labels = tf.gather(labels, order)

    order_shuffled = tf.argsort(shuffled_labels)
    shuffled_images = tf.gather(shuffled_images, order_shuffled)
    shuffled_labels = tf.gather(shuffled_labels, order_shuffled)

    pair_labels = tf.cast((shuffled_labels == labels), tf.float32)

    norm = tf.cast((pair_labels == pair_labels), tf.float32)
    return images, shuffled_images, pair_labels, norm


def pair_augmentation(images, labels):
    pos_images, pos_shuffled_images, pos_pair_labels, pos_norm = positive_augmentation(
        images, labels
    )
    neg_images, neg_shuffled_images, neg_pair_labels, neg_norm = negative_pairs(
        images, labels
    )
    images = tf.concat([pos_images, neg_images], axis=0)
    shuffled_images = tf.concat([pos_shuffled_images, neg_shuffled_images], axis=0)
    pair_labels = tf.concat([pos_pair_labels, neg_pair_labels], axis=0)
    norm = tf.concat([pos_norm, neg_norm], axis=0)
    return {"image_a": images, "image_b": shuffled_images}, {
        "similarity": pair_labels,
        "norm_a": norm,
        "norm_b": norm,
    }


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
    print(f"Batch images shape: {images['image_a'].shape}")
    print(f"Batch labels shape: {labels['similarity'].shape}")

leaky_relu_layer = layers.LeakyReLU(negative_slope=0.1)


# Define the CNN encoder model
def make_encoder(latent_dim):
    model = models.Sequential()

    # Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (5, 5)))
    model.add(leaky_relu_layer)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (5, 5)))
    model.add(leaky_relu_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3)))
    model.add(leaky_relu_layer)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dim))
    model.add(leaky_relu_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(latent_dim))
    model.add(leaky_relu_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(latent_dim))
    model.add(leaky_relu_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(latent_dim, activation="linear"))
    return model


class CosineSimilarityLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarityLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Split inputs into two tensors
        vector_a, vector_b = inputs

        # Compute the dot product of the vectors

        # Compute the L2 norms (magnitudes) of each vector
        norm_a = tf.sqrt(tf.reduce_sum(tf.square(vector_a), axis=-1))
        norm_b = tf.sqrt(tf.reduce_sum(tf.square(vector_b), axis=-1))

        # Compute cosine similarity
        dot_product = tf.reduce_sum(tf.multiply(vector_a, vector_b), axis=-1)
        cosine_similarity = dot_product / (norm_a * norm_b + tf.keras.backend.epsilon())

        return cosine_similarity, norm_a, norm_b


def make_model(encoder, cos_value):
    relu_layer = tf.keras.layers.ReLU(
        max_value=1,
        negative_slope=0,
        threshold=cos_value,
    )
    # Define the two input layers
    input_a = layers.Input(shape=(28, 28, 1))
    input_a = data_augmentation(input_a)
    input_b = layers.Input(shape=(28, 28, 1))
    input_b = data_augmentation(input_b)

    # Get the encoded representations of the inputs
    encoded_a = encoder(input_a)
    encoded_b = encoder(input_b)

    # Compute the cosine similarity between the encoded vectors
    cosine_similarity, norm_a, norm_b = CosineSimilarityLayer()([encoded_a, encoded_b])

    # reshaped_similarity = relu_layer(cosine_similarity)
    reshaped_similarity = cosine_similarity

    # Create the model
    model = models.Model(
        inputs={"image_a": input_a, "image_b": input_b},
        outputs={"similarity": reshaped_similarity, "norm_a": norm_a, "norm_b": norm_b},
    )

    return model


if __name__ == "__main__":
    # Example: Cosine of 45 degrees (converted to radians)
    latent_dim = 256
    n_clusters = 10
    encoder = make_encoder(latent_dim)
    encoder.summary()
    degree = min(latent_dim * 180 / n_clusters, 90)
    angle_in_radians = math.radians(degree)  # Convert degrees to radians
    cos_value = math.cos(angle_in_radians)
    model = make_model(encoder, cos_value)

    model.summary()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.01,
        clipvalue=1.0,
        momentum=0.95,  # Momentum helps in smoothing out the updates
        nesterov=True,  # Nesterov momentum is often used in deep learning for better convergence)
    )
    model.compile(
        optimizer=optimizer,
        loss={"similarity": "mae", "norm_a": "mae", "norm_b": "mae"},
        loss_weights={"similarity": 1, "norm_a": 1, "norm_b": 1},
        # metrics={"similarity": "mae", "norm_a": "mae", "norm_b": "mae"},
    )

    model.fit(train_dataset, epochs=50, validation_data=encoder_test_dataset)

    encoder.save("encoder.keras")
