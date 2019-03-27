import tensorflow as tf
import numpy as np
import skimage
import pathlib


class GAN(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):
        # =========================================================================================
        real_images, labels = real_input_fn()
        fake_latents = fake_input_fn()
        training = tf.placeholder(tf.bool)
        # =========================================================================================
        fake_images = generator(fake_latents, labels, training)
        # =========================================================================================
        real_logits = discriminator(real_images, labels, training)
        fake_logits = discriminator(fake_images, labels, training)
        # =========================================================================================
        generator_losses = tf.nn.softplus(-fake_logits)
        # -----------------------------------------------------------------------------------------
        # gradient-based mode-seeking loss
        latent_gradients = tf.gradients(fake_images, [fake_latents])[0]
        mode_seeking_losses = 1.0 / (tf.reduce_sum(tf.square(latent_gradients), axis=[1]) + 1e-6)
        generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight
        # -----------------------------------------------------------------------------------------
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits)
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
        self.fake_images = tf.transpose(fake_images, [0, 2, 3, 1])
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
                            fake_images=self.fake_images
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

    def generate(self, model_dir, data_dir, config):

        real_data_dir = pathlib.Path(data_dir) / "reals"
        fake_data_dir = pathlib.Path(data_dir) / "fakes"

        if not real_data_dir.exists():
            real_data_dir.mkdir(parents=True, exist_ok=True)
        if not fake_data_dir.exists():
            fake_data_dir.mkdir(parents=True, exist_ok=True)

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

            def unnormalize(inputs, mean, std):
                return inputs * std + mean

            def generator():

                while True:
                    try:
                        yield session.run(
                            fetches=[self.real_images, self.fake_images],
                            feed_dict={self.training: False}
                        )
                    except tf.errors.OutOfRangeError:
                        break

            for i, (real_image, fake_image) in enumerate(zip(*map(np.concatenate, zip(*generator())))):
                skimage.io.imsave(real_data_dir / "{}.jpg".format(i), unnormalize(real_image, 0.5, 0.5))
                skimage.io.imsave(fake_data_dir / "{}.jpg".format(i), unnormalize(fake_image, 0.5, 0.5))
