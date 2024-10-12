from src.data.mnist import get_encoder_dataset
from src.model.encoder import get_models
from tensorflow.keras import callbacks
import tensorflow as tf

# tf.data.experimental.enable_debug_mode()
import os

batch_size = 256
latent_dim = 64
train_dataset, test_dataset, input_shape = get_encoder_dataset(batch_size)
encoder, encoder_train_model, _, _ = get_models(input_shape, latent_dim=latent_dim)
print(encoder.summary())
print(encoder_train_model.summary())
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True,
    mode="min",
)


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, func):
        super(SaveModelCallback, self).__init__()
        self.func = func

    def on_epoch_end(self, epoch, logs=None):
        self.func()


def save_model():
    encoder.save("encoder.keras")
    encoder_train_model.save("encoder_train_model.keras")
    print("Model saved")


encoder_train_model.fit(
    train_dataset,
    epochs=1000,
    validation_data=test_dataset,
    callbacks=[early_stopping, SaveModelCallback(save_model)],
)
