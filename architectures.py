import numpy as np
import tensorflow as tf


def lrelu(x):
    with tf.name_scope("LReLu"):
        return tf.maximum(x, 0.2 * x)


def make_translation_generator(layers):
    def generator(image):
        # Encoder
        hidden = image
        for layer in range(layers):
            hidden = tf.layers.conv2d(hidden, 64, kernel_size=4, strides=2, activation=lrelu, padding="SAME")

        # Decoder
        for layer in range(layers-1):
            hidden = tf.layers.conv2d_transpose(hidden, 64, kernel_size=4, strides=2, activation=tf.nn.relu,
                                                padding="SAME")

        new_image = tf.layers.conv2d_transpose(hidden, 3, kernel_size=4, strides=2, activation=tf.nn.tanh,
                                               padding="SAME")
        return new_image

    return generator


def make_discriminator(layers):
    def discriminator(image):
        hidden = image
        for layer in range(layers):
            hidden = tf.layers.conv2d(hidden, 32 * 2**layer, kernel_size=4, strides=2, activation=lrelu)

        shape = hidden.shape
        total = np.prod(shape.as_list()[1:])
        flat = tf.reshape(hidden, (-1, total))
        final = tf.layers.dense(flat, 1)
        return final

    return discriminator