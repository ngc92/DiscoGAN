import time

import tensorflow as tf

from disco.gan import disco_gan, DeviceMapping
from disco.input import input_fn, make_preprocessor
from disco.models import make_translation_generator, make_discriminator

generator = make_translation_generator(4)
disc_fn = make_discriminator(4)
preprocess = make_preprocessor(256, 64)

p1 = "trainA/*.jpg"
p2 = "trainB/*.jpg"

with tf.Graph().as_default():
    with tf.device("/cpu:0"):
        train_disco = disco_gan(input_fn(p1, preprocess), input_fn(p2, preprocess), generator, disc_fn,
                                DeviceMapping("/cpu:0", "/gpu:0", "gpu:1", "gpu:0", "gpu:1"))

    with tf.train.MonitoredTrainingSession(checkpoint_dir="test", save_summaries_steps=10,
                                           config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        while True:
            t = time.time()
            sess.run(train_disco)
            print(time.time() - t)
