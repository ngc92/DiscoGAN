import time
import os
import argparse
import tensorflow as tf
import numpy as np
import scipy.misc

from disco.gan import disco_gan, DeviceMapping
from disco.input import input_pipeline, convert_image, crop_and_resize_image, augment_with_flips, \
    augment_with_rotations, thicken, random_crop, augment_contrast
from disco.models import make_deconv_generator, make_discriminator, make_unet_generator, make_upsample_generator

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="", type=str)
parser.add_argument("--checkpoint-dir", default="ckpt", type=str)
parser.add_argument("--A", default="trainA", type=str)
parser.add_argument("--B", default="trainB", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--curriculum", default=1000, type=int)
parser.add_argument("--input-threads", default=2, type=int)
parser.add_argument("--image-size", default=64, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--GPUs", default=1, type=int)
parser.add_argument("--generator-depth", default=3, type=int)
parser.add_argument("--discriminator-depth", default=3, type=int)
parser.add_argument("--summary-interval", default=100, type=int)
# eval mode
parser.add_argument("--eval", action='store_true')
parser.add_argument("--out-dir", default="result", type=str)

args = parser.parse_args()

generator_ab = make_upsample_generator(args.generator_depth, data_format="channels_first", channels=1)
generator_ba = make_upsample_generator(args.generator_depth, data_format="channels_first", channels=1)
#generator = make_unet_generator(args.generator_depth, 32, data_format="channels_first")
discriminator = make_discriminator(args.discriminator_depth, data_format="channels_first")

if args.GPUs == 0:
    devices = DeviceMapping("/cpu:0", "/cpu:0", "/cpu:0", "/cpu:0", "/cpu:0")
elif args.GPUs == 1:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:0", "/gpu:0", "/gpu:0")
elif args.GPUs == 2:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:1", "/gpu:0", "/gpu:1")
elif args.GPUs == 4:
    devices = DeviceMapping("/cpu:0", "/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3")


pA = os.path.join(args.data_dir, args.A, "*")
pB = os.path.join(args.data_dir, args.B, "*")


if args.eval:
    cell_input_fn = input_pipeline(pA, crop_and_resize_image("min", args.image_size),
                                   num_threads=args.input_threads, epochs=1, batch_size=args.batch_size, greyscale=True)
    seg_input_fn = input_pipeline(pB, crop_and_resize_image("min", args.image_size),
                                  num_threads=args.input_threads, epochs=1, batch_size=args.batch_size, greyscale=True)

    # THIS IS EXTREMELY UGLY
    # GET PYTHON 3 WORKING AND WE CAN REMOVE IT
    try:
        os.makedirs(os.path.dirname(os.path.join(args.out_dir, args.A)), exist_ok=True)
    except: pass
    try:
        os.makedirs(os.path.dirname(os.path.join(args.out_dir, args.B)), exist_ok=True)
    except: pass

    with tf.Graph().as_default():
        train_disco = disco_gan(cell_input_fn, seg_input_fn, devices, args.curriculum, discriminator=discriminator,
                                generator_AB=generator_ab, generator_BA=generator_ba, is_training=False)
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
                    scipy.misc.imsave(fna, np.squeeze(fB))

                    fnB = os.path.relpath(fnB.decode(), start=args.data_dir)
                    fnb = os.path.join(args.out_dir, fnB)
                    scipy.misc.imsave(fnb, np.squeeze(fA))

else:
    preprocess = random_crop(512, 64) | crop_and_resize_image("min", args.image_size) | \
                 augment_with_flips(vertical=True) | augment_with_rotations()

    # augmentation with contrast is problemnatic, because we cannot recreate it from the
    # segmentations.
    cell_input_fn = input_pipeline(pA, preprocess, num_threads=args.input_threads,
                                   epochs=args.epochs, batch_size=args.batch_size, greyscale=True)
    seg_input_fn = input_pipeline(pB, thicken() | preprocess, num_threads=args.input_threads, epochs=args.epochs,
                                  batch_size=args.batch_size, greyscale=True)

    with tf.Graph().as_default():
        train_disco = disco_gan(cell_input_fn, seg_input_fn, devices, args.curriculum, discriminator,
                                generator_AB=generator_ab, generator_BA=generator_ba)
        saver = tf.train.Saver()

        with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint_dir,
                                               save_summaries_steps=args.summary_interval,
                                               config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            while not sess.should_stop():
                t = time.time()
                sess.run(train_disco.train_step)
                print(time.time() - t)
