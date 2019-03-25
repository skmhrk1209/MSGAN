import tensorflow as tf
import numpy as np
import pickle
import os
import glob
from utils import Struct


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


def cifar10_input_fn(filenames, batch_size, num_epochs, shuffle):

    def unpickle(file):
        with open(file, "rb") as file:
            dict = pickle.load(file, encoding="bytes")
        return dict

    dicts = [unpickle(filename) for filename in filenames]
    images = np.concatenate([dict[b"data"] for dict in dicts])
    labels = np.concatenate([dict[b"labels"] for dict in dicts])

    def preprocess(images, labels):

        images = tf.reshape(images, [-1, 3, 32, 32])
        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.image.random_flip_left_right(images)
        images = linear_map(images, 0.0, 1.0, -1.0, 1.0)

        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, 10)

        return images, labels

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(images),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(
        map_func=preprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
