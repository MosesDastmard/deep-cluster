import tensorflow as tf
import numpy as np


class SiameseGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size=32, batch_num=1000, negative=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.negative = negative

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        # take the first batch_size random images
        idxs = np.random.choice(len(self.images), self.batch_size)
        batch_images1 = self.images[idxs]
        batch_labels1 = self.labels[idxs]

        # take the second batch_size random images
        idxs = np.random.choice(len(self.images), self.batch_size)
        batch_images2 = self.images[idxs]
        batch_labels2 = self.labels[idxs]

        # make concatenated images

        # take the second batch_size random images
        idxs = np.random.choice(len(self.images), self.batch_size)
        sorted_images1 = self.images[idxs]
        sorted_labels1 = self.labels[idxs]

        # take the second batch_size random images
        idxs = np.random.choice(len(self.images), self.batch_size)
        sorted_images2 = self.images[idxs]
        sorted_labels2 = self.labels[idxs]

        idxs = np.argsort(sorted_labels1)
        sorted_labels1 = sorted_labels1[idxs]
        sorted_images1 = sorted_images1[idxs]

        idxs = np.argsort(sorted_labels2)
        sorted_labels2 = sorted_labels2[idxs]
        sorted_images2 = sorted_images2[idxs]

        images_a = np.concatenate([batch_images1, sorted_images1], axis=0)
        images_b = np.concatenate([batch_images2, sorted_images2], axis=0)
        labels_a = np.concatenate([batch_labels1, sorted_labels1], axis=0)
        labels_b = np.concatenate([batch_labels2, sorted_labels2], axis=0)

        labels = np.where(labels_a == labels_b, 0, 1)
        if self.negative:
            labels = np.where(labels_a == labels_b, 1, 1)
        else:
            labels = np.where(labels_a == labels_b, 0, 1)
        return {"images_a": images_a, "images_b": images_b}, labels
