import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from utils import Struct


class GAN(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):
        # =========================================================================================
        real_images, labels = real_input_fn()
        fake_latents1 = fake_input_fn()
        fake_latents2 = fake_input_fn()
        training = tf.placeholder(tf.bool)
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
        # distance-based mode-seeking loss
        latent_distances = tf.reduce_sum(tf.abs(fake_latents1 - fake_latents2), axis=[1])
        image_distances = tf.reduce_sum(tf.abs(fake_images1 - fake_images2), axis=[1, 2, 3])
        mode_seeking_losses = latent_distances / (image_distances + 1e-6)
        generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight
        '''
        # gradient-based mode-seeking loss
        latent_gradients = tf.gradients(fake_images1, [fake_latents1])[0]
        mode_seeking_losses = 1.0 / (tf.reduce_sum(tf.square(latent_gradients), axis=[1]) + 1e-6)
        generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight
        '''
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
        self.training = training
        self.real_images = tf.transpose(real_images, [0, 2, 3, 1])
        self.fake_images1 = tf.transpose(fake_images1, [0, 2, 3, 1])
        self.fake_images2 = tf.transpose(fake_images2, [0, 2, 3, 1])
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.discriminator_train_op = discriminator_train_op
        self.generator_train_op = generator_train_op

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
                    summary_op=tf.summary.merge(list(map(
                        lambda name_tensor: tf.summary.image(*name_tensor), dict(
                            real_images=self.real_images,
                            fake_images1=self.fake_images1,
                            fake_images2=self.fake_images2
                        ).items()
                    )))
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge(list(map(
                        lambda name_tensor: tf.summary.scalar(*name_tensor), dict(
                            discriminator_loss=self.discriminator_loss,
                            generator_loss=self.generator_loss
                        ).items()
                    )))
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        discriminator_loss=self.discriminator_loss,
                        generator_loss=self.generator_loss
                    ),
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                session.run(self.discriminator_train_op, feed_dict={self.training: True})
                session.run(self.generator_train_op, feed_dict={self.training: True})

    def evaluate(self, model_dir, config):

        real_features = tf.contrib.gan.eval.run_inception(
            images=tf.contrib.gan.eval.preprocess_image(self.real_images),
            output_tensor="pool_3:0"
        )
        fake_features = tf.contrib.gan.eval.run_inception(
            images=tf.contrib.gan.eval.preprocess_image(self.fake_images1),
            output_tensor="pool_3:0"
        )

        def generator():
            while True:
                try:
                    yield session.run(
                        fetches=[real_features, fake_features],
                        feed_dict={self.training: False}
                    )
                except tf.errors.OutOfRangeError:
                    break

        all_real_features = tf.placeholder(tf.float32)
        all_fake_features = tf.placeholder(tf.float32)
        frechet_inception_distance = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(
            real_activations=all_real_features,
            generated_activations=all_fake_features
        )

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

            frechet_inception_distance = session.run(
                fetches=frechet_inception_distance,
                feed_dict=dict(zip(
                    [all_real_features, all_fake_features],
                    map(np.concatenate, zip(*generator()))
                ))
            )
            tf.logging.info("frechet_inception_distance: {}".format(frechet_inception_distance))
