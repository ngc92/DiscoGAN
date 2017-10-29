import argparse

import tensorflow as tf
import numpy as np


def inscope(function, scope):
    reuse = {"value": False}
    def f(*args, **kwargs):
        with tf.variable_scope(scope, reuse=reuse["value"]):
            reuse["value"] = True
            return function(*args, **kwargs)
    return f


def edm_gan(input, encoder, decoder, transformer, discriminator):
    with tf.device("/cpu:0"):
        A, B = input()
        tf.summary.image("A", A)
        tf.summary.image("B", B)

    # scoped functions
    encoder = inscope(encoder, "Encoder")
    decoder = inscope(decoder, "Decoder")

    discriminator_a = inscope(discriminator, "CA")
    discriminator_b = inscope(discriminator, "CB")

    transformer_ab = inscope(transformer, "TAB")
    transformer_ba = inscope(transformer, "TBA")

    # auto-encoding
    eA = encoder(A)
    eB = encoder(B)

    aeA_l = _autoencoder_loss(eA, A, decoder)
    aeB_l = _autoencoder_loss(eB, B, decoder)
    tf.summary.scalar("loss/autoencode_A", aeA_l)
    tf.summary.scalar("loss/autoencode_A", aeB_l)

    # transformer
    fB = decoder(transformer_ab(eA))
    fA = decoder(transformer_ba(eB))
    tf.summary.image("fake_A", fA)
    tf.summary.image("fake_B", fB)

    # discrimination
    with tf.device("/gpu:1"):
        dA_l, gA_l = _discriminate_fake_loss(fA, discriminator_a)
        dB_l, gB_l = _discriminate_fake_loss(fB, discriminator_b)

    tf.summary.scalar("loss/d_fake_A", dA_l)
    tf.summary.scalar("loss/d_fake_B", dB_l)
    tf.summary.scalar("loss/generator_A", gA_l)
    tf.summary.scalar("loss/generator_B", gB_l)

    # reconstruction
    reB = transformer_ab(encoder(fA))
    reA = transformer_ba(encoder(fB))
    tf.summary.image("reconstructed_A", decoder(reA))
    tf.summary.image("reconstructed_B", decoder(reB))

    # reconstruction error
    rB_l = tf.losses.mean_squared_error(reB, eB)
    rA_l = tf.losses.mean_squared_error(reA, eA)
    tf.summary.scalar("loss/rec_A", rA_l)
    tf.summary.scalar("loss/rec_B", rB_l)

    # combine the losses for the generative part
    auto_enc_loss = aeA_l + aeB_l
    rec_loss = rB_l + rA_l
    fool_loss = gA_l + gB_l

    generator_loss = 0.1 * auto_enc_loss + 0.1 * rec_loss + fool_loss

    # the losses for the discriminator
    dfake_loss = dA_l + dB_l
    with tf.device("/gpu:1"):
        drA, _ = discriminator_a(A)
        drB, _ = discriminator_b(B)
        drA_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drA), drA)
        drB_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drB), drB)

    tf.summary.scalar("loss/d_real_A", drA_l)
    tf.summary.scalar("loss/d_real_B", drB_l)

    dreal_loss = drA_l + drB_l

    discriminator_loss = dfake_loss + dreal_loss

    tf.summary.scalar("loss/generator", generator_loss)
    tf.summary.scalar("loss/discriminator", discriminator_loss)

    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Encoder") + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Decoder") + \
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
    tf.summary.image("decoded", decoded)
    return tf.losses.mean_squared_error(decoded, original, reduction=tf.losses.Reduction.MEAN)


def _discriminate_fake_loss(fake, discriminator):
    logits, _ = discriminator(fake)
    d_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits), logits, reduction=tf.losses.Reduction.MEAN)
    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits), logits, reduction=tf.losses.Reduction.MEAN)

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
        return tf.layers.conv2d(representation, representation.shape[3], 1, 1, padding="SAME", use_bias=True)

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
from disco.input import input_pipeline, convert_image, crop_and_resize_image, augment_with_flips
import os


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="", type=str)
parser.add_argument("--checkpoint-dir", default="ckpt", type=str)
parser.add_argument("--A", default="trainA/*", type=str)
parser.add_argument("--B", default="trainB/*", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--curriculum", default=1000, type=int)
parser.add_argument("--input-threads", default=2, type=int)
parser.add_argument("--image-size", default=64, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--GPUs", default=1, type=int)
parser.add_argument("--generator-depth", default=3, type=int)
parser.add_argument("--discriminator-depth", default=3, type=int)
parser.add_argument("--summary-interval", default=100, type=int)

args = parser.parse_args()

preprocess = crop_and_resize_image("min", 64) | augment_with_flips() | convert_image()

pA = os.path.join(args.data_dir, args.A)
pB = os.path.join(args.data_dir, args.B)


def input_fn(p1, p2):
    def f():
        return input_pipeline(p1, preprocess, num_threads=4, epochs=1000, batch_size=32)()[0], \
               input_pipeline(p2, preprocess, num_threads=4, epochs=1000, batch_size=32)()[0]
    return f


with tf.Graph().as_default():
    tg, td = edm_gan(input_fn(pA, pB), make_encoder(3, 2), make_decoder(3), make_transformer(),
                     make_discriminator(3))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint_dir,
                                           save_summaries_steps=args.summary_interval,
                                           config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(100000):
            if i % 4 == 0:
                sess.run(td)
            else:
                sess.run(tg)
