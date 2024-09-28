import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from src.model.util import L2NormLayer


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
    images_sorted = tf.gather(images, order)
    labels_sorted = tf.gather(labels, order)

    order_shuffled = tf.argsort(shuffled_labels)
    shuffled_images_sorted = tf.gather(shuffled_images, order_shuffled)
    shuffled_labels_sorted = tf.gather(shuffled_labels, order_shuffled)

    pair_labels = tf.cast((shuffled_labels_sorted == labels_sorted), tf.float32)
    return images_sorted, shuffled_images_sorted, pair_labels


def pair_augmentation(images, labels):
    shuffled_images, shuffled_labels = shuffle__images(images, labels)
    pos_images, pos_shuffled_images, pos_pair_labels = positive_pairs__(
        images, labels, shuffled_images, shuffled_labels
    )
    neg_images, neg_shuffled_images, neg_pair_labels = negative_pairs__(
        images, labels, shuffled_images, shuffled_labels
    )
    images_a = tf.concat([pos_images, neg_images], axis=0)
    images_b = tf.concat([pos_shuffled_images, neg_shuffled_images], axis=0)
    pair_labels = tf.concat([pos_pair_labels, neg_pair_labels], axis=0)
    return {
        "images_a": images_a,
        "images_b": images_b,
    }, {
        "dense": pair_labels,
        "similarity": pair_labels,
    }


def disc_augmentation(vectors):
    n_samples = tf.shape(vectors)[0]
    pos_labels = tf.ones((n_samples,), dtype=tf.float32)
    neg_labels = tf.zeros((n_samples,), dtype=tf.float32)
    stddev = np.random.uniform(0.0001, 1)
    maskprob = np.random.uniform(0.0001, 1)
    mask = tf.cast(
        tf.random.uniform(shape=tf.shape(vectors)) < maskprob, dtype=tf.float32
    )
    noisy_vecotrs = vectors + mask * tf.random.normal(
        shape=tf.shape(vectors), mean=0.0, stddev=stddev
    )
    tanh_vector = tf.clip_by_value(noisy_vecotrs, -1, 1)
    labels = tf.concat([pos_labels, neg_labels], axis=0)
    features = tf.concat([vectors, tanh_vector], axis=0)
    return features, labels


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
    ],
    name="data_augmentation",
)


# Preprocessing and batching for training dataset
def get_encoder_dataset__(train_dataset, test_dataset, input_shape, batch_size):
    similarity_dataset = (
        dataset.shuffle(4 * batch_size)
        .batch(batch_size)
        .map(
            lambda images, labels: pair_augmentation(images, labels),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
        for dataset in [train_dataset, test_dataset]
    )
    return *similarity_dataset, input_shape


def get_discriminator_dataset__(
    encoder, train_dataset, test_dataset, input_shape, batch_size
):
    discriminator_dataset = (
        dataset.shuffle(4 * batch_size)
        .batch(batch_size)
        .map(
            lambda images, labels: encoder(images),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda images: disc_augmentation(images),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
        for dataset in [train_dataset, test_dataset]
    )
    return *discriminator_dataset, input_shape
