from src.data.mnist import get_discriminator_dataset
from src.model.discriminator import get_model
from tensorflow.keras import callbacks
from tensorflow.keras import models

batch_size = 32

encoder = models.load_model("encoder.keras")
train_dataset, test_dataset, input_shape = get_discriminator_dataset(
    encoder, batch_size
)
discriminator = get_model(encoder)
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True,
    mode="min",
)
discriminator.fit(
    train_dataset,
    epochs=500,
    validation_data=test_dataset,
    callbacks=[early_stopping],
)
discriminator.save("discriminator.keras")
