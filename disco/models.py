import numpy as np
import tensorflow as tf
from numbers import Integral

"""
  Properties of the original DiscoGan Architecture:
   * They disable BIAS in all convolutional layers
   * The discriminator has only convolutional layers. 
     But the last layer is just a linear layer nonetheless, due to no padding
   * Generator: LReLU in encoder, ReLU in decoder.
   * BatchNorm before nonlinearity.
"""


def lrelu(x):
    with tf.name_scope("LReLu"):
        return tf.maximum(x, 0.2 * x)


def make_translation_generator(layers, channels=3, stride=2, encoding_noise=None, shortcuts=False):
    def generator(image, is_training=True):
        # Encoder
        hidden = image
        encodings = []

        with tf.variable_scope("encoder"):
            for layer in range(layers):
                hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=stride,
                                          padding="SAME", use_bias=False, name="conv%i" % layer)
                if layer > 0:
                    hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer)

                # apply the nonlinearity after batch-norm. No idea if this is relevant
                hidden = lrelu(hidden)
                encodings += [hidden]

        # add some noise to the encoding
        # TODO figure out whether this makes sense
        if encoding_noise is not None and encoding_noise > 0:
            in_shape = hidden.shape.as_list()
            noise = tf.random_normal([tf.shape(hidden)[0], in_shape[1], in_shape[2], encoding_noise])
            hidden = tf.concat([hidden, noise], axis=3)

        # Decoder
        with tf.variable_scope("decoder"):
            for layer in range(layers-1):
                if layer > 0 and shortcuts:
                    hidden = tf.concat([hidden, encodings[layers - layer - 1]], axis=3)
                hidden = tf.layers.conv2d_transpose(hidden, 64 * 2**(layers - layer - 2), kernel_size=4, strides=stride,
                                                    padding="SAME", use_bias=False, name="deconv%i" % layer)
                hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer)
                hidden = tf.nn.relu(hidden)

            if shortcuts:
                hidden = tf.concat([hidden, encodings[0]], axis=3)
            new_image = tf.layers.conv2d_transpose(hidden, channels, kernel_size=4, strides=stride, activation=tf.nn.tanh,
                                                   padding="SAME", use_bias=False, name="deconv%i" % layers)

        return new_image

    return generator


def make_discriminator(layers):
    def discriminator(image, is_training=True):
        hidden = image
        features = []
        for layer in range(layers):
            hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=2, name="conv%i" % layer)
            if layer > 0:
                hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer)

            hidden = lrelu(hidden)
            features += [hidden]

        shape = hidden.shape
        total = np.prod(shape.as_list()[1:])
        flat = tf.reshape(hidden, (-1, total))
        final = tf.layers.dense(flat, 1, use_bias=False)
        return final, features[1:]

    return discriminator
