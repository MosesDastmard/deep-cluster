from tensorflow.keras import layers, models
import tensorflow as tf
from src.data.util import data_augmentation
from src.model.util import L2NormLayer, SimilarityLayer

relu_layer = layers.ReLU(negative_slope=0.1)


# Define the CNN encoder model
def make_encoder(input_shape, latent_dim):
    model = models.Sequential(name="encoder")
    # Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (5, 5)))
    model.add(relu_layer)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5)))
    model.add(relu_layer)
    model.add(layers.Conv2D(128, (3, 3)))
    model.add(relu_layer)
    model.add(layers.Flatten())
    model.add(layers.Dense(4 * latent_dim, activation="relu"))
    model.add(layers.Dense(2 * latent_dim, activation="relu"))
    model.add(layers.Dense(1 * latent_dim, activation="tanh"))
    return model


def mask_mse_loss(y_true, y_pred):
    mask = tf.cast(y_true == 1, tf.float32)
    return tf.reduce_mean(tf.square(mask * (y_true - y_pred)))


def mask_mae_loss(y_true, y_pred):
    mask = tf.cast(y_true == 1, tf.float32)
    return tf.reduce_mean(tf.abs(mask * (y_true - y_pred)))


def make_model(encoder):
    input_a = layers.Input(shape=encoder.input_shape[1:], name="images_a")
    input_b = layers.Input(shape=encoder.input_shape[1:], name="images_b")

    input_a_ = data_augmentation(input_a)
    input_b_ = data_augmentation(input_b)

    # Get the encoded representations of the inputs
    encoded_a = encoder(input_a_)
    encoded_b = encoder(input_b_)

    # Compute the absolute difference between the two feature vectors
    abs_layer = tf.keras.layers.Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]), name="abs_diff"
    )
    abs_diff = abs_layer([encoded_a, encoded_b])
    output = layers.Dense(1, activation="sigmoid", name="dense")(abs_diff)
    similarity_layer = SimilarityLayer(name="similarity")
    similarity = similarity_layer([encoded_a, encoded_b])
    model = models.Model(
        inputs={
            "images_a": input_a,
            "images_b": input_b,
        },
        outputs={"dense": output, "similarity": similarity},
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=1e-3,
        momentum=0.95,  # Momentum helps in smoothing out the updates
        nesterov=True,  # Nesterov momentum is often used in deep learning for better convergence)
    )
    model.compile(
        optimizer=optimizer,
        loss={
            "dense": "binary_crossentropy",
            "similarity": mask_mae_loss,
        },
        loss_weights={
            "dense": 1.0,
            "similarity": 5.0,
        },
        metrics={
            "dense": ["accuracy"],  # Add accuracy for the 'dense' output
            "similarity": ["mae"],  # Add mae for the 'similarity' output
        },
    )

    return model


def get_models(input_shape, latent_dim=64):
    encoder = make_encoder(input_shape, latent_dim)
    model = make_model(encoder)
    return encoder, model
