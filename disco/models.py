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


def _image_format(image, data_format):
    if data_format == "channels_first":
        channel_index = 1
        image = tf.transpose(image, [0, 3, 1, 2])
    else:
        channel_index = 3
    return image, channel_index


def make_deconv_generator(layers, channels=3, stride=2, data_format="channels_last"):
    # allow layer dependent stride but if just one is given, use for all layers
    if not isinstance(stride, list):
        stride = [stride] * layers

    def generator(image, is_training=True):
        image, channel_index = _image_format(image, data_format)

        # Encoder
        hidden = image

        with tf.variable_scope("encoder"):
            for layer in range(layers):
                hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=stride[layer],
                                          padding="SAME", use_bias=False, name="conv%i" % layer,
                                          data_format=data_format)
                if layer > 0:
                    hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer,
                                                           axis=channel_index)

                # apply the nonlinearity after batch-norm. No idea if this is relevant
                hidden = tf.nn.leaky_relu(hidden)

        # Decoder
        with tf.variable_scope("decoder"):
            for layer in range(layers-1):
                index = layers - layer - 1
                hidden = tf.layers.conv2d_transpose(hidden, 64 * 2**(layers - layer - 2), kernel_size=4,
                                                    strides=stride[index], padding="SAME", use_bias=False,
                                                    name="deconv%i" % layer, data_format=data_format)
                hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer,
                                                       axis=channel_index)
                hidden = tf.nn.relu(hidden)

            new_image = tf.layers.conv2d_transpose(hidden, channels, kernel_size=4, strides=stride[0],
                                                   activation=tf.nn.tanh, padding="SAME",
                                                   use_bias=False, name="deconv%i" % layers, data_format=data_format)

        if data_format == "channels_first":
            new_image = tf.transpose(new_image, [0, 2, 3, 1])

        return new_image

    return generator


def make_upsample_generator(layers, channels=3, stride=2, data_format="channels_last"):
    # allow layer dependent stride but if just one is given, use for all layers
    if not isinstance(stride, list):
        stride = [stride] * layers

    def generator(image, is_training=True):
        image, channel_index = _image_format(image, data_format)

        # Encoder
        hidden = image

        with tf.variable_scope("encoder"):
            for layer in range(layers):
                hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=stride[layer],
                                          padding="SAME", use_bias=False, name="conv%i" % layer,
                                          data_format=data_format)
                if layer > 0:
                    hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer,
                                                           axis=channel_index)

                # apply the nonlinearity after batch-norm. No idea if this is relevant
                hidden = tf.nn.leaky_relu(hidden)

        # Decoder
        with tf.variable_scope("decoder"):
            for layer in range(layers-1):
                index = layers - layer - 1
                if data_format == "channels_last":
                    size = (stride[index]*hidden.shape.as_list()[1], stride[index]*hidden.shape.as_list()[2])
                    hidden = tf.image.resize_images(hidden, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                else:
                    size = (stride[index] * hidden.shape.as_list()[2], stride[index] * hidden.shape.as_list()[3])
                    hidden = tf.transpose(hidden, [0, 2, 3, 1])
                    hidden = tf.image.resize_images(hidden, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    hidden = tf.transpose(hidden, [0, 3, 1, 2])

                hidden = tf.layers.conv2d(hidden, 64 * 2**(layers - layer - 2), use_bias=False, padding="same",
                                          kernel_size=4, strides=1, data_format=data_format)
                hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer,
                                                       axis=channel_index)
                hidden = tf.nn.relu(hidden)

            new_image = tf.layers.conv2d_transpose(hidden, channels, kernel_size=4, strides=stride[0],
                                                   activation=tf.nn.tanh, padding="SAME",
                                                   use_bias=False, name="deconv%i" % layers, data_format=data_format)

        if data_format == "channels_first":
            new_image = tf.transpose(new_image, [0, 2, 3, 1])

        return new_image

    return generator


def make_unet_generator(layers, features, data_format="channels_last"):
    def unet(image, is_training=True):
        image, channel_index = _image_format(image, data_format)

        def block_down(input, features, data_format):
            conv1 = tf.layers.conv2d(input, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
            conv2 = tf.layers.conv2d(conv1, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
            pool = tf.layers.max_pooling2d(conv2, 2, 2, data_format=data_format)
            return conv2, pool

        def block_up(input, cat, features, data_format):
            # deconv upscaling
            up = tf.layers.conv2d_transpose(input, features, 2, strides=2, activation=tf.nn.relu, padding="same",
                                            data_format=data_format)
            # concat
            up = tf.concat([up, cat], axis=1 if data_format == "channels_first" else 3)

            conv1 = tf.layers.conv2d(up, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
            conv2 = tf.layers.conv2d(conv1, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
            return conv2

        intermediate = []
        hidden = image
        for i in range(layers):
            conv, hidden = block_down(hidden, 2**i * features, data_format)
            intermediate += [conv]

        hidden, _ = block_down(hidden, 2 ** layers * features, data_format)

        for i in range(layers-1, -1, -1):
            hidden = block_up(hidden, intermediate[i], 2**i * features, data_format)

        result = tf.layers.conv2d(hidden, image.shape.as_list()[channel_index], 1, activation=tf.nn.tanh, padding="same",
                                  data_format=data_format)

        if data_format == "channels_first":
            result = tf.transpose(result, [0, 2, 3, 1])

        return result

    return unet


def make_discriminator(layers, data_format="channels_last"):
    def discriminator(image, is_training=True):
        image, channel_index = _image_format(image, data_format)

        hidden = image
        features = []
        for layer in range(layers):
            hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=2, name="conv%i" % layer,
                                      data_format=data_format)
            if layer > 0:
                hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer,
                                                       axis=channel_index)

            hidden = tf.nn.leaky_relu(hidden)
            features += [hidden]

        flat = tf.layers.flatten(hidden)
        final = tf.layers.dense(flat, 1, use_bias=False)
        return final, features[1:]

    return discriminator
