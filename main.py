import time
import argparse
import tensorflow as tf

from disco.gan import disco_gan, DeviceMapping
from disco.input import input_pipeline, convert_image, crop_and_resize_image, augment_with_flips
from disco.models import make_translation_generator, make_discriminator

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--dataA", default="trainA/*")
parser.add_argument("--dataB", default="trainB/*")
parser.add_argument("--epochs", default=100)
parser.add_argument("--input-threads", default=1)
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

with tf.Graph().as_default():
    with tf.device("/cpu:0"):
        train_disco = disco_gan(input_fn(args.dataA), input_fn(args.dataB), generator, discriminator, devices)

    with tf.train.MonitoredTrainingSession(checkpoint_dir="test", save_summaries_steps=25,
                                           config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        while True:
            t = time.time()
            sess.run(train_disco)
            print(time.time() - t)
