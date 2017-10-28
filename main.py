import time
import os
import argparse
import tensorflow as tf
import scipy.misc

from disco.gan import disco_gan, DeviceMapping
from disco.input import input_pipeline, convert_image, crop_and_resize_image, augment_with_flips
from disco.models import make_translation_generator, make_discriminator

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="", type=str)
parser.add_argument("--checkpoint-dir", default="ckpt", type=str)
parser.add_argument("--A", default="trainA/*", type=str)
parser.add_argument("--B", default="trainB/*", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--input-threads", default=2, type=int)
parser.add_argument("--image-size", default=64, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--GPUs", default=1, type=int)
parser.add_argument("--generator-depth", default=3, type=int)
parser.add_argument("--discriminator-depth", default=3, type=int)
# eval mode
parser.add_argument("--eval", action='store_true')
parser.add_argument("--out-dir", default="result", type=str)

args = parser.parse_args()

generator_AB = make_translation_generator(args.generator_depth)
generator_BA = make_translation_generator(args.generator_depth, encoding_noise=10)
discriminator = make_discriminator(args.discriminator_depth)
preprocess = crop_and_resize_image("min", args.image_size) | augment_with_flips() | convert_image()

if args.GPUs == 0:
    devices = DeviceMapping("/cpu:0", "/cpu:0", "/cpu:0", "/cpu:0", "/cpu:0")
elif args.GPUs == 1:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:0", "/gpu:0", "/gpu:0")
elif args.GPUs == 2:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:1", "/gpu:0", "/gpu:1")
elif args.GPUs == 4:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3")


pA = os.path.join(args.data_dir, args.A)
pB = os.path.join(args.data_dir, args.B)

if args.eval:
    def input_fn(path):
        return input_pipeline(path, crop_and_resize_image("min", args.image_size) | convert_image(),
                              num_threads=args.input_threads, epochs=1, batch_size=args.batch_size)


    # THIS IS EXTREMELY UGLY
    # GET PYTHON 3 WORKING AND WE CAN REMOVE IT
    try:
        os.makedirs(os.path.dirname(os.path.join(args.out_dir, args.A)), exist_ok=True)
    except: pass
    try:
        os.makedirs(os.path.dirname(os.path.join(args.out_dir, args.B)), exist_ok=True)
    except: pass

    with tf.Graph().as_default():
        train_disco = disco_gan(input_fn(pA), input_fn(pB), devices, 1000, discriminator=discriminator,
                                is_training=False, generator_AB=generator_AB, generator_BA=generator_BA)
        saver = tf.train.Saver()

        with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(
                config=tf.ConfigProto(allow_soft_placement=True))) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
            while True:
                t = time.time()
                fA, fB, fnA, fnB = sess.run([train_disco.fakeA, train_disco.fakeB, train_disco.file_name_A,
                                             train_disco.file_name_B])

                for fA, fB, fnA, fnB in zip(fA, fB, fnA, fnB):
                    fnA = os.path.relpath(fnA.decode(), start=args.data_dir)
                    fna = os.path.join(args.out_dir, fnA)
                    scipy.misc.imsave(fna, fB)

                    fnB = os.path.relpath(fnB.decode(), start=args.data_dir)
                    fnb = os.path.join(args.out_dir, fnB)
                    scipy.misc.imsave(fnb, fA)

else:
    def input_fn(path):
        return input_pipeline(path, preprocess, num_threads=args.input_threads, epochs=args.epochs,
                              batch_size=args.batch_size)

    with tf.Graph().as_default():
        train_disco = disco_gan(input_fn(pA), input_fn(pB), devices, 1000, discriminator=discriminator,
                                generator_AB=generator_AB, generator_BA=generator_BA)
        saver = tf.train.Saver()

        with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint_dir, save_summaries_steps=25,
                                               config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            while True:
                t = time.time()
                sess.run(train_disco.train_step)
                print(time.time() - t)
