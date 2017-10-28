from collections import namedtuple
from functools import wraps

import tensorflow as tf


DiscoGan = namedtuple("DiscoGan", ["train_step", "realA", "fakeA", "realB", "fakeB", "file_name_A", "file_name_B"])


def disco_gan(input_A, input_B, generator, discriminator, device_mapping, curriculum, is_training=True, generator_AB=None,
              generator_BA=None, discriminator_A=None, discriminator_B=None):
    if generator_AB is None:
        generator_AB = generator

    if generator_BA is None:
        generator_BA = generator

    if discriminator_A is None:
        discriminator_A = discriminator

    if discriminator_B is None:
        discriminator_B = discriminator

    # now, wrap the discriminator to ensure consistent output
    def wrap(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            result = f(*args, **kwargs)
            if not isinstance(result, tuple):
                return result, []
            return result
        return wrapped

    return _disco_gan(input_A, input_B, generator_AB, generator_BA, wrap(discriminator_A),
                      wrap(discriminator_B), device_mapping, curriculum, is_training)


def _feature_matching(fake, real, scope="feature_matching"):
    with tf.name_scope(scope):
        losses = []
        for f, r in zip(fake, real):
            mf = tf.reduce_mean(f, axis=0)
            mr = tf.reduce_mean(r, axis=0)
            loss = tf.losses.mean_squared_error(mr, mf, reduction=tf.losses.Reduction.MEAN)
            losses += [loss]
        return losses


def _disco_gan(input_A, input_B, generator_AB, generator_BA, discriminator_A, discriminator_B, device_mapping,
               curriculum, is_training):
    # create and summarize the inputs
    with tf.device(device_mapping.input):
        A, file_A = input_A()
        B, file_B = input_B()
        tf.summary.image("A", A)
        tf.summary.image("B", B)

    # create and summarize fakes
    with tf.device(device_mapping.genA), tf.variable_scope("genA"):
        fA = generator_BA(B, is_training)
        tf.summary.image("fA", fA)

    with tf.device(device_mapping.genB), tf.variable_scope("genB"):
        fB = generator_AB(A, is_training)
        tf.summary.image("fB", fB)

    # reconstruction
    with tf.device(device_mapping.genA), tf.variable_scope("genA", reuse=True):
        rA = generator_BA(fB, is_training)
        tf.summary.image("rA", rA)

    with tf.device(device_mapping.genB), tf.variable_scope("genB", reuse=True):
        rB = generator_AB(fA, is_training)
        tf.summary.image("rB", rB)

    # discriminators
    with tf.device(device_mapping.disA), tf.variable_scope("disA"):
        dfA, dffA = discriminator_A(fA, is_training)

    with tf.device(device_mapping.disA), tf.variable_scope("disA", reuse=True):
        drA, drfA = discriminator_A(A, is_training)

    with tf.device(device_mapping.disB), tf.variable_scope("disB"):
        dfB, dffB = discriminator_B(fB, is_training)

    with tf.device(device_mapping.disB), tf.variable_scope("disB", reuse=True):
        drB, drfB = discriminator_B(B, is_training)

    # now all the loss terms
    with tf.name_scope("DA_loss"):
        dfA_l = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dfA), logits=dfA, scope="fake_loss",
                                                reduction=tf.losses.Reduction.MEAN)
        drA_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drA), drA, scope="real_loss",
                                                reduction=tf.losses.Reduction.MEAN, label_smoothing=0.1)
        loss_dA = drA_l + dfA_l

    with tf.name_scope("DB_loss"):
        dfB_l = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dfB), dfB, scope="fake_loss",
                                                reduction=tf.losses.Reduction.MEAN)
        drB_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drB), drB, scope="real_loss",
                                                reduction=tf.losses.Reduction.MEAN, label_smoothing=0.1)
        loss_dB = drB_l + dfB_l

    rate = tf.cond(tf.greater(tf.train.get_or_create_global_step(), curriculum),
                   lambda: tf.constant(0.5), lambda: tf.constant(0.01))

    loss_AB = _generator_loss(dfB, dffB, drfB, B, rB, rate, "GAB_loss")
    loss_BA = _generator_loss(dfA, dffA, drfA, A, rA, rate, "GBA_loss")

    tf.summary.scalar("loss_dA", loss_dA)
    tf.summary.scalar("loss_dB", loss_dB)

    tf.summary.histogram("dfA", dfA)
    tf.summary.histogram("drA", drA)
    tf.summary.histogram("dfB", dfB)
    tf.summary.histogram("drB", drB)

    var_GAB = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "genB")
    var_GBA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "genA")

    var_DA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disA")
    var_DB = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disB")

    optimizer_GAB = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    optimizer_GBA = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    optimizer_DA = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999)
    optimizer_DB = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999)

    opt_AB = optimizer_GAB.minimize(loss_AB, var_list=var_GAB, colocate_gradients_with_ops=True)
    opt_BA = optimizer_GBA.minimize(loss_BA, var_list=var_GBA, colocate_gradients_with_ops=True)
    opt_DA = optimizer_DA.minimize(loss_dA, var_list=var_DA, colocate_gradients_with_ops=True)
    opt_DB = optimizer_DB.minimize(loss_dB, var_list=var_DB, global_step=tf.train.get_or_create_global_step(),
                                   colocate_gradients_with_ops=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.group(opt_AB, opt_BA, opt_DA, opt_DB)

    return DiscoGan(train_step=train_step, realA=A, realB=B, fakeA=fA, fakeB=fB, file_name_A=file_A, file_name_B=file_B)


def _generator_loss(discriminator_logit, fake_features, real_features, real, reconstructed, rate, scope):
    with tf.name_scope(scope):
        # discrimination loss
        discrimination = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_logit), discriminator_logit,
                                                         scope="discriminate", reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar("discrimination", discrimination)

        # feature matching loss
        feature_matching = tf.add_n(_feature_matching(fake_features, real_features, "matching"))
        tf.summary.scalar("feature_matching", feature_matching)

        # reconstruction loss
        reconstruction = tf.losses.mean_squared_error(real, reconstructed, reduction=tf.losses.Reduction.MEAN,
                                                      scope="reconstruct")
        tf.summary.scalar("reconstruction", reconstruction)

        total = (feature_matching + 0.1*discrimination) * (1.0 - rate) + rate * reconstruction
        tf.summary.scalar("total", total)
    return total


DeviceMapping = namedtuple("DeviceMapping", ("input", "genA", "genB", "disA", "disB"))