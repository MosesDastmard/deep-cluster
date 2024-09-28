from sklearn import svm
import numpy as np
from src.data.mnist import train_dataset, test_dataset
from src.data.fashion import test_dataset as fashion_test_dataset
from tensorflow.keras.models import load_model

encoder = load_model("encoder.keras")
X_train = encoder.predict(train_dataset.map(lambda x, y: x).batch(32))

# Train the One-Class SVM
model = svm.OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)
model.fit(X_train)

# Testing with new data (some anomalies)
Xmnist_test = encoder.predict(test_dataset.map(lambda x, y: x).batch(32))
Xfashion_test = encoder.predict(fashion_test_dataset.map(lambda x, y: x).batch(32))
X_test = np.concatenate([Xmnist_test, Xfashion_test])
labels = np.concatenate(
    [np.ones(Xmnist_test.shape[0]), -np.ones(Xfashion_test.shape[0])]
)
predictions = model.predict(X_test)
print((predictions == labels).mean())


discriminator = load_model("discriminator.keras")
predictions = discriminator.predict(X_test)
labels = np.concatenate(
    [np.ones(Xmnist_test.shape[0]), np.zeros(Xfashion_test.shape[0])]
)
print(((predictions > 0.75).astype(int).flatten() == labels).mean())
