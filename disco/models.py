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


def make_translation_generator(layers, channels=3, stride=2):
    # allow layer dependent stride but if just one is given, use for all layers
    if not isinstance(stride, list):
        stride = [stride] * layers

    def generator(image, is_training=True):
        # Encoder
        hidden = image

        with tf.variable_scope("encoder"):
            for layer in range(layers):
                hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=stride[layer],
                                          padding="SAME", use_bias=False, name="conv%i" % layer)
                if layer > 0:
                    hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer)

                # apply the nonlinearity after batch-norm. No idea if this is relevant
                hidden = tf.nn.leaky_relu(hidden)

        # Decoder
        with tf.variable_scope("decoder"):
            for layer in range(layers-1):
                index = layers - layer - 1
                hidden = tf.layers.conv2d_transpose(hidden, 64 * 2**(layers - layer - 2), kernel_size=4,
                                                    strides=stride[index], padding="SAME", use_bias=False,
                                                    name="deconv%i" % layer)
                hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer)
                hidden = tf.nn.relu(hidden)

            new_image = tf.layers.conv2d_transpose(hidden, channels, kernel_size=4, strides=stride[0],
                                                   activation=tf.nn.tanh, padding="SAME",
                                                   use_bias=False, name="deconv%i" % layers)

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

            hidden = tf.nn.leaky_relu(hidden)
            features += [hidden]

        flat = tf.layers.flatten(hidden)
        final = tf.layers.dense(flat, 1, use_bias=False)
        return final, features[1:]

    return discriminator
