#=================================================================================================#
# TensorFlow implementation of MSGAN
# [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis]
# (https://arxiv.org/pdf/1903.05628v1.pdf)
#=================================================================================================#

import tensorflow as tf
import argparse
import functools
import glob
from dataset import cifar10_input_fn
from model import GAN
from network import DCGAN
from utils import Struct

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="cifar10_msgan_model")
parser.add_argument('--filenames', type=str, default="cifar-10-batches-py/data_batch_*")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--total_steps", type=int, default=100000)
parser.add_argument('--train', action="store_true")
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--generate', action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    dcgan = DCGAN(
        min_resolution=[4, 4],
        max_resolution=[32, 32],
        min_channels=128,
        max_channels=512
    )

    gan = GAN(
        generator=dcgan.generator,
        discriminator=dcgan.discriminator,
        real_input_fn=functools.partial(
            cifar10_input_fn,
            filenames=glob.glob(args.filenames),
            batch_size=args.batch_size,
            num_epochs=args.num_epochs if args.train else 1,
            shuffle=True if args.train else False,
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 100])
        ),
        hyper_params=Struct(
            generator_learning_rate=2e-4,
            generator_beta1=0.5,
            generator_beta2=0.999,
            discriminator_learning_rate=2e-4,
            discriminator_beta1=0.5,
            discriminator_beta2=0.999,
            mode_seeking_loss_weight=1.0,
        )
    )

    if args.train:
        gan.train(
            model_dir=args.model_dir,
            total_steps=args.total_steps,
            save_checkpoint_steps=10000,
            save_summary_steps=1000,
            log_tensor_steps=1000,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )

    if args.evaluate:
        gan.evaluate(
            model_dir=args.model_dir,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )

    if args.generate:
        gan.generate(
            model_dir=args.model_dir,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
