import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import skimage
import metrics
from utils import Struct


class GAN(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):
        # =========================================================================================
        real_images, labels = real_input_fn()
        fake_latents1 = fake_input_fn()
        fake_latents2 = fake_input_fn()
        training = tf.placeholder(tf.bool, [])
        # =========================================================================================
        fake_images1 = generator(fake_latents1, labels, training)
        fake_images2 = generator(fake_latents2, labels, training)
        # =========================================================================================
        real_logits = discriminator(real_images, labels, training)
        fake_logits1 = discriminator(fake_images1, labels, training)
        fake_logits2 = discriminator(fake_images2, labels, training)
        # =========================================================================================
        generator_losses = tf.nn.softplus(-fake_logits1)
        generator_losses += tf.nn.softplus(-fake_logits2)
        generator_losses /= 2
        # -----------------------------------------------------------------------------------------
        # gradient-based mode-seeking loss
        latent_gradients = tf.gradients(fake_images1, [fake_latents1])[0]
        mode_seeking_losses = 1.0 / (tf.reduce_sum(tf.square(latent_gradients), axis=[1]) + 1e-6)
        generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight
        # -----------------------------------------------------------------------------------------
        discriminator_losses = tf.nn.softplus(fake_logits1)
        discriminator_losses += tf.nn.softplus(fake_logits2)
        discriminator_losses /= 2
        discriminator_losses += tf.nn.softplus(-real_logits)
        # -----------------------------------------------------------------------------------------
        generator_loss = tf.reduce_mean(generator_losses)
        discriminator_loss = tf.reduce_mean(discriminator_losses)
        # =========================================================================================
        generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.generator_learning_rate,
            beta1=hyper_params.generator_beta1,
            beta2=hyper_params.generator_beta2
        )
        discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.discriminator_learning_rate,
            beta1=hyper_params.discriminator_beta1,
            beta2=hyper_params.discriminator_beta2
        )
        # -----------------------------------------------------------------------------------------
        generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        # =========================================================================================
        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        # tensors and operations used later
        self.operations = Struct(
            discriminator_train_op=discriminator_train_op,
            generator_train_op=generator_train_op
        )
        self.placeholders = Struct(
            training=training
        )
        self.tensors = Struct(
            global_step=tf.train.get_global_step(),
            real_images=tf.transpose(real_images, [0, 2, 3, 1]),
            fake_images1=tf.transpose(fake_images1, [0, 2, 3, 1]),
            fake_images2=tf.transpose(fake_images2, [0, 2, 3, 1]),
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss
        )

    def train(self, model_dir, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    )
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.ndims == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items()
                    ])
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if tensor.shape.ndims == 0
                    },
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                for name, operation in self.operations.items():
                    session.run(
                        fetches=operation,
                        feed_dict={self.placeholders.training: True}
                    )

    def evaluate(self, model_dir, config):

        inception = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
        image_size = hub.get_expected_image_size(inception)

        real_features = inception(tf.image.resize_images(self.tensors.real_images, image_size))
        fake_features = inception(tf.image.resize_images(self.tensors.fake_images1, image_size))

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            def generator():
                while True:
                    try:
                        yield session.run(
                            fetches=[real_features, fake_features],
                            feed_dict={self.placeholders.training: False}
                        )
                    except tf.errors.OutOfRangeError:
                        break

            frechet_inception_distance = metrics.frechet_inception_distance(*map(np.concatenate, zip(*generator())))
            tf.logging.info("frechet_inception_distance: {}".format(frechet_inception_distance))
