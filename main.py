import time
import os
import argparse
import tensorflow as tf

from disco.gan import disco_gan, DeviceMapping
from disco.input import input_pipeline, convert_image, crop_and_resize_image, augment_with_flips
from disco.models import make_translation_generator, make_discriminator

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="ckpt")
parser.add_argument("--checkpoint-dir", default="")
parser.add_argument("--A", default="trainA/*")
parser.add_argument("--B", default="trainB/*")
parser.add_argument("--epochs", default=100)
parser.add_argument("--input-threads", default=2)
parser.add_argument("--image-size", default=64)
parser.add_argument("--batch-size", default=32)
parser.add_argument("--GPUs", default=1)
args = parser.parse_args()

generator = make_translation_generator(4)
discriminator = make_discriminator(4)
preprocess = crop_and_resize_image("min", args.image_size) | augment_with_flips() | convert_image()

if args.GPUs == 0:
    devices = DeviceMapping("/cpu:0", "/cpu:0", "/cpu:0", "/cpu:0", "/cpu:0")
elif args.GPUs == 1:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:0", "/gpu:0", "/gpu:0")
elif args.GPUs == 2:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:1", "/gpu:0", "/gpu:1")
elif args.GPUs == 4:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3")


def input_fn(path):
    return input_pipeline(path, preprocess, num_threads=args.input_threads, epochs=args.epochs,
                          batch_size=args.batch_size)

pA = os.path.join(args.data_dir, args.A)
pB = os.path.join(args.data_dir, args.B)

with tf.Graph().as_default():
    train_disco = disco_gan(input_fn(pA), input_fn(pB), generator, discriminator, devices)

    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint_dir, save_summaries_steps=25,
                                           config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        while True:
            t = time.time()
            sess.run(train_disco)
            print(time.time() - t)
