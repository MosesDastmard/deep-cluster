from src.data.mnist import get_encoder_dataset
from src.model.encoder import get_models
from tensorflow.keras import callbacks

batch_size = 512
latent_dim = 64
train_dataset, test_dataset, input_shape = get_encoder_dataset(batch_size)
encoder, model = get_models(input_shape, latent_dim=latent_dim)
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True,
    mode="min",
)
model.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset,
    callbacks=[early_stopping],
)
encoder.save("encoder.keras")
