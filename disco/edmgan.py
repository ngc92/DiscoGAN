import tensorflow as tf
import numpy as np


def inscope(function, scope):
    reuse = {"value": False}
    def f(*args, **kwargs):
        with tf.variable_scope(scope, reuse=reuse["value"]):
            return function(*args, **kwargs)
    return f


def edm_gan(input, encoder, decoder, transformer, discriminator):
    with tf.device("/cpu:0"):
        A, B = input()

    # scoped functions
    encoder_a = inscope(encoder, "EA")
    encoder_b = inscope(encoder, "EB")

    decoder_a = inscope(decoder, "DA")
    decoder_b = inscope(decoder, "DB")

    discriminator_a = inscope(discriminator, "CA")
    discriminator_b = inscope(discriminator, "CB")

    transformer_ab = inscope(transformer, "TAB")
    transformer_ba = inscope(transformer, "TBA")

    # auto-encoding
    eA = encoder_a(A)
    eB = encoder_b(B)

    aeA_l = _autoencoder_loss(eA, A, decoder_a)
    aeB_l = _autoencoder_loss(eB, B, decoder_b)

    # transformer
    fB = decoder_b(transformer_ab(eA))

    fA = decoder_a(transformer_ba(eB))

    # discrimination
    dA_l, gA_l = _discriminate_fake_loss(fA, discriminator_a)
    dB_l, gB_l = _discriminate_fake_loss(fB, discriminator_b)

    # reconstruction
    reB = transformer_ab(encoder_a(fA))
    reA = transformer_ba(encoder_b(fB))

    # reconstruction error
    rB_l = tf.losses.mean_squared_error(reB, eB)
    rA_l = tf.losses.mean_squared_error(reA, eA)


    # combine the losses for the generative part
    auto_enc_loss = aeA_l + aeB_l
    rec_loss = rB_l + rA_l
    fool_loss = gA_l + gB_l

    generator_loss = auto_enc_loss + rec_loss + fool_loss

    # the losses for the discriminator
    dfake_loss = dA_l + dB_l
    drA = discriminator_a(A)
    drB = discriminator_b(B)
    dreal_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(drA), drA) + \
                 tf.losses.sigmoid_cross_entropy(tf.ones_like(drB), drB)

    discriminator_loss = dfake_loss + dreal_loss

    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "EA") + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "EB") + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DA") + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DB") + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TAB") + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TBA")

    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CA") + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CB")

    # optimizers
    optimizer_G = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    optimizer_D = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999)

    train_G = optimizer_G.minimize(generator_loss, global_step=tf.train.get_or_create_global_step(),
                                   var_list=generator_vars, colocate_gradients_with_ops=True)

    train_D = optimizer_D.minimize(discriminator_loss, global_step=tf.train.get_or_create_global_step(),
                                   var_list=discriminator_vars, colocate_gradients_with_ops=True)

    return train_G, train_D


def _autoencoder_loss(encoded, original, decoder):
    decoded = decoder(encoded)
    return tf.losses.mean_squared_error(decoded, original, reduction=tf.losses.Reduction.MEAN)


def _discriminate_fake_loss(fake, discriminator):
    logits = discriminator(fake)
    d_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits), logits)
    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits), logits)

    return d_loss, g_loss


########################################################################################################################

def lrelu(x):
    with tf.name_scope("LReLu"):
        return tf.maximum(x, 0.2 * x)


def make_decoder(layers, channels=3, stride=2, ):
    # allow layer dependent stride but if just one is given, use for all layers
    if not isinstance(stride, list):
        stride = [stride] * layers

    def generator(image, is_training=True):
        # Encoder
        hidden = image

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


def make_encoder(layers, stride=2):
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
                hidden = lrelu(hidden)
        return hidden

    return generator


def make_transformer():
    def generator(representation, is_training=True):
        shape = representation.shape
        total = np.prod(shape.as_list()[1:])
        flat = tf.reshape(representation, (-1, total))
        transformed = tf.layers.dense(flat, flat.shape[1])
        return tf.reshape(transformed, shape)

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


########################################################################################################################
from .input import input_pipeline, convert_image, crop_and_resize_image, augment_with_flips
import os
preprocess = crop_and_resize_image("min", 64) | augment_with_flips() | convert_image()

pA = os.path.join(args.data_dir, args.A)
pB = os.path.join(args.data_dir, args.B)

def input_fn(p1, p2):
    return input_pipeline(p1, preprocess, num_threads=4, epochs=1000, batch_size=32), \
           input_pipeline(p2, preprocess, num_threads=4, epochs=1000, batch_size=32)


with tf.Graph().as_default():
    tg, td = edm_gan(input_fn(pA, pB), make_encoder(3, 2), make_decoder(3), make_transformer(),
                          make_discriminator(3))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint_dir,
                                           save_summaries_steps=args.summary_interval,
                                           config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(100000):
            if i % 3 == 0:
                sess.run(train_disco.train_step)
            else:
                sess.run(train_disco.train_step)