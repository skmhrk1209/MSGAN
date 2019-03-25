import tensorflow as tf
import numpy as np


class DCGAN(object):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels):

        self.min_resolution = np.asanyarray(min_resolution)
        self.max_resolution = np.asanyarray(max_resolution)
        self.min_channels = min_channels
        self.max_channels = max_channels

        def log2(x): return 0 if (x == 1).all() else 1 + log2(x >> 1)
        self.depth = log2(self.max_resolution // self.min_resolution)

    def generator(self, latents, labels, training, name="generator", reuse=tf.AUTO_REUSE):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.depth - depth))

        with tf.variable_scope(name, reuse=reuse):

            inputs = tf.concat([latents, labels], axis=1)
            inputs = inputs[..., tf.newaxis, tf.newaxis]

            for depth in range(self.depth + 1):
                if depth:
                    inputs = tf.layers.conv2d_transpose(
                        inputs=inputs,
                        filters=channels(depth),
                        kernel_size=[4, 4],
                        strides=[2, 2],
                        padding="same",
                        data_format="channels_first",
                        use_bias=True,
                        kernel_initializer=tf.initializers.he_normal(),
                        bias_initializer=tf.initializers.zeros()
                    )
                    inputs = tf.layers.batch_normalization(
                        inputs=inputs,
                        axis=1,
                        training=training
                    )
                    inputs = tf.nn.relu(inputs)
                else:
                    inputs = tf.layers.conv2d_transpose(
                        inputs=inputs,
                        filters=channels(depth),
                        kernel_size=[4, 4],
                        strides=[1, 1],
                        padding="valid",
                        data_format="channels_first",
                        use_bias=True,
                        kernel_initializer=tf.initializers.he_normal(),
                        bias_initializer=tf.initializers.zeros()
                    )
                    inputs = tf.layers.batch_normalization(
                        inputs=inputs,
                        axis=1,
                        training=training
                    )
                    inputs = tf.nn.relu(inputs)

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=3,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
                data_format="channels_first",
                use_bias=True,
                kernel_initializer=tf.initializers.he_normal(),
                bias_initializer=tf.initializers.zeros()
            )
            inputs = tf.nn.tanh(inputs)

            return inputs

    def discriminator(self, images, labels, training, name="discriminator", reuse=tf.AUTO_REUSE):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.depth - depth))

        with tf.variable_scope(name, reuse=reuse):

            labels = labels[..., tf.newaxis, tf.newaxis]
            labels = tf.tile(labels, [1, 1, *images.shape[2:]])
            inputs = tf.concat([images, labels], axis=1)

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=channels(self.depth),
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
                data_format="channels_first",
                use_bias=True,
                kernel_initializer=tf.initializers.he_normal(),
                bias_initializer=tf.initializers.zeros()
            )
            inputs = tf.nn.leaky_relu(inputs)

            for depth in range(self.depth + 1)[::-1]:
                if depth:
                    inputs = tf.layers.conv2d(
                        inputs=inputs,
                        filters=channels(depth),
                        kernel_size=[4, 4],
                        strides=[2, 2],
                        padding="same",
                        data_format="channels_first",
                        use_bias=True,
                        kernel_initializer=tf.initializers.he_normal(),
                        bias_initializer=tf.initializers.zeros()
                    )
                    inputs = tf.layers.batch_normalization(
                        inputs=inputs,
                        axis=1,
                        training=training
                    )
                    inputs = tf.nn.leaky_relu(inputs)
                else:
                    inputs = tf.layers.conv2d(
                        inputs=inputs,
                        filters=1,
                        kernel_size=[4, 4],
                        strides=[1, 1],
                        padding="valid",
                        data_format="channels_first",
                        use_bias=True,
                        kernel_initializer=tf.initializers.he_normal(),
                        bias_initializer=tf.initializers.zeros()
                    )
                    inputs = tf.squeeze(inputs)

            return inputs
