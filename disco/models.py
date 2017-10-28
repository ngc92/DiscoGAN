import numpy as np
import tensorflow as tf

"""
  Properties of the original DiscoGan Architecture:
   * They disable BIAS in all convolutional layers
   * The discriminator has only convolutional layers. 
     But the last layer is just a linear layer nonetheless, due to no padding
   * Generator: LReLU in encoder, ReLU in decoder.
   * BatchNorm after nonlinearity.
"""


def lrelu(x):
    with tf.name_scope("LReLu"):
        return tf.maximum(x, 0.2 * x)


def make_translation_generator(layers):
    def generator(image):
        # Encoder
        hidden = image
        for layer in range(layers):
            hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=2, activation=lrelu, padding="SAME",
                                      use_bias=False)
            if layer > 0:
                hidden = tf.layers.batch_normalization(hidden, training=True)

        # Decoder
        for layer in range(layers-1):
            hidden = tf.layers.conv2d_transpose(hidden, 64 * 2**(layers - layer - 2), kernel_size=4, strides=2,
                                                activation=tf.nn.relu, padding="SAME", use_bias=False)
            hidden = tf.layers.batch_normalization(hidden, training=True)

        new_image = tf.layers.conv2d_transpose(hidden, 3, kernel_size=4, strides=2, activation=tf.nn.tanh,
                                               padding="SAME", use_bias=False)
        return new_image

    return generator


def make_discriminator(layers):
    def discriminator(image):
        hidden = image
        features = []
        for layer in range(layers):
            hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=2, activation=lrelu)
            features += [hidden]
            if layer > 0:
                hidden = tf.layers.batch_normalization(hidden, training=True)

        shape = hidden.shape
        total = np.prod(shape.as_list()[1:])
        flat = tf.reshape(hidden, (-1, total))
        final = tf.layers.dense(flat, 1, use_bias=False)
        return final, features

    return discriminator
