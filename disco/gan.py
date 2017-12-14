from collections import namedtuple
from functools import wraps

import tensorflow as tf


DiscoGan = namedtuple("DiscoGan", ["train_step", "realA", "fakeA", "realB", "fakeB", "file_name_A", "file_name_B"])


def disco_gan(input_A, input_B, device_mapping, curriculum, discriminator=None, generator=None, is_training=True,
              generator_AB=None, generator_BA=None, discriminator_A=None, discriminator_B=None):
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


def _hierarchical_reconstruction_matching(real, reconstructed, scales, name=None, weights=1.0):
    assert scales >= 1
    with tf.name_scope(name, "hierarchical_reconstruction_loss", [real, reconstructed]):
        # reconstruction loss
        losses = []
        for i in range(scales):
            if i != 0:
                reconstructed = tf.layers.average_pooling2d(reconstructed, 2, 2)
                real = tf.layers.average_pooling2d(real, 2, 2)
            r_loss = tf.losses.mean_squared_error(real, reconstructed, reduction=tf.losses.Reduction.MEAN,
                                                  scope="reconstruct_%i" % i, weights=weights)
            losses.append(r_loss)

        return tf.add_n(losses) / float(scales)


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
    fA = tf.identity(fA, "fake_A")

    with tf.device(device_mapping.genB), tf.variable_scope("genB"):
        fB = generator_AB(A, is_training)
        tf.summary.image("fB", fB)
    fB = tf.identity(fB, "fake_B")

    # reconstruction
    with tf.device(device_mapping.genA), tf.variable_scope("genA", reuse=True):
        rA = generator_BA(fB, is_training)
        tf.summary.image("rA", rA)
    rA = tf.identity(rA, "reconstructed_A")

    with tf.device(device_mapping.genB), tf.variable_scope("genB", reuse=True):
        rB = generator_AB(fA, is_training)
        tf.summary.image("rB", rB)
    rB = tf.identity(rB, "reconstructed_B")

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
    loss_dA = _discriminator_loss(dfA, drA, "DA_loss")
    loss_dB = _discriminator_loss(dfB, drB, "DB_loss")

    rate = tf.cond(tf.greater(tf.train.get_or_create_global_step(), curriculum),
                   lambda: tf.constant(0.5), lambda: tf.constant(0.01))

    loss_AB, lab_dic = _generator_loss(dfB, dffB, drfB, B, rB, rate, "GAB_loss")
    loss_BA, lba_dic = _generator_loss(dfA, dffA, drfA, A, rA, rate, "GBA_loss")

    # gradient summaries
    _generator_gradient_summary(lab_dic, fA, fB, rB, "gradient_AB")
    _generator_gradient_summary(lba_dic, fB, fA, rA, "gradient_BA")

    tf.summary.histogram("dfA", dfA)
    tf.summary.histogram("drA", drA)
    tf.summary.histogram("dfB", dfB)
    tf.summary.histogram("drB", drB)

    var_GAB = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "genB")
    var_GBA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "genA")

    var_DA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disA")
    var_DB = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disB")

    optimizer_G = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.5, beta2=0.999)
    optimizer_D = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999)

    # Their code on GitHub uses only two distinct optimizers.
    def optimize_generator():
        return optimizer_G.minimize(loss_AB + loss_BA, var_list=var_GAB + var_GBA, colocate_gradients_with_ops=True)

    def optimize_discriminator():
        return optimizer_D.minimize(loss_dB + loss_dA, var_list=var_DB + var_DA, colocate_gradients_with_ops=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        discriminator_step = tf.equal(tf.train.get_or_create_global_step() % 3, 0)
        train_step = tf.cond(discriminator_step, optimize_discriminator, optimize_generator)
        with tf.control_dependencies([train_step]):
            train_step = tf.train.get_global_step().assign_add(1)

    return DiscoGan(train_step=train_step, realA=A, realB=B, fakeA=fA, fakeB=fB, file_name_A=file_A, file_name_B=file_B)


def _discriminator_loss(logits_fake, logits_real, scope):
    with tf.name_scope(scope):
        fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits=logits_fake, scope="fake_loss",
                                                    reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar("fake", fake_loss)
        tf.summary.scalar("fake_p", tf.reduce_mean(tf.nn.sigmoid(logits_fake)))

        real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real, scope="real_loss",
                                                    reduction=tf.losses.Reduction.MEAN, label_smoothing=0.1)
        tf.summary.scalar("real", real_loss)
        tf.summary.scalar("real_p", tf.reduce_mean(tf.nn.sigmoid(logits_real)))

        total = real_loss + fake_loss
        tf.summary.scalar("total", total)
    return total


def _generator_loss(discriminator_logit, fake_features, real_features, real, reconstructed, rate, scope):
    with tf.name_scope(scope):
        # discrimination loss
        discrimination = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_logit), discriminator_logit,
                                                         scope="discriminate", reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar("discrimination", discrimination)

        # fake_p
        # these will be considered fixed values, do not propagate gradient
        fake_p = tf.nn.sigmoid(tf.stop_gradient(discriminator_logit))
        fake_p = tf.expand_dims(tf.expand_dims(fake_p, 2), 3)
        per_example_weights = tf.maximum(1.0, 3.0*fake_p) / tf.cast(tf.shape(discriminator_logit)[0], tf.float32)

        # feature matching loss
        feature_matching = tf.add_n(_feature_matching(fake_features, real_features, "matching"))
        tf.summary.scalar("feature_matching", feature_matching)

        # reconstruction loss
        reconstruction = _hierarchical_reconstruction_matching(real, reconstructed, 3, "reconstruct",
                                                               weights=per_example_weights)
        tf.summary.scalar("reconstruction", reconstruction)

        discrimination_loss = (feature_matching + 0.1*discrimination)

        rate = rate * tf.reduce_mean(per_example_weights)
        tf.summary.scalar("rate", rate)

        total = discrimination_loss * (1.0 - rate) + rate * reconstruction
        tf.summary.scalar("total", total)
    return total, {"reconstruction": reconstruction, "discriminate": discrimination_loss}


def _generator_gradient_summary(losses, other_fake, fake, reconstructed, name):
    reconstruction = losses["reconstruction"]
    discriminate = losses["discriminate"]
    with tf.name_scope(name):
        grad_r = tf.gradients(reconstruction, other_fake)[0]
        grad_f = tf.gradients(reconstruction, reconstructed)[0]
        grad_d = tf.gradients(discriminate, fake)[0]
        tf.summary.image("rec", grad_r)
        tf.summary.image("rloss", grad_f)
        tf.summary.image("dis", grad_d)
        tf.summary.scalar("norm_rec", tf.nn.l2_loss(grad_r))
        tf.summary.scalar("norm_dis", tf.nn.l2_loss(grad_d))
        tf.summary.scalar("norm_rgrad", tf.nn.l2_loss(grad_f))


# TODO this does not work yet
def unet_gan(input, generator, discriminator, device_mapping, curriculum, is_training):
    # create and summarize the inputs
    with tf.device(device_mapping.input):
        A, B, file_A, file_B = input()
        tf.summary.image("A", A)
        tf.summary.image("B", B)

    # create and summarize fakes
    with tf.device(device_mapping.genB), tf.variable_scope("Generator"):
        fB = generator(A, is_training)
        tf.summary.image("fB", fB)

    # discriminators
    with tf.device(device_mapping.disB), tf.variable_scope("Discriminator"):
        dfB, dffB = discriminator(fB, is_training)

    with tf.device(device_mapping.disB), tf.variable_scope("Discriminator", reuse=True):
        drB, drfB = discriminator(B, is_training)

    # now all the loss terms
    discriminator_loss = _discriminator_loss(dfB, drB, "DB_loss")
    generator_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dfB), dfB, scope="discriminate",
                                                     reduction=tf.losses.Reduction.MEAN)

    # reconstruction loss
    reconstruction_loss = tf.losses.mean_squared_error(B, fB, scope="reconstruction_loss")

    tf.summary.histogram("dfB", dfB)
    tf.summary.histogram("drB", drB)

    var_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator")
    var_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator")

    optimizer_G = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    optimizer_D = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999)

    # Their code on GitHub uses only two distinct optimizers.
    def optimize_generator():
        return optimizer_G.minimize(generator_loss + reconstruction_loss, var_list=var_gen,
                                    colocate_gradients_with_ops=True)

    def optimize_discriminator():
        return optimizer_D.minimize(discriminator_loss, var_list=var_dis)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.cond(tf.equal(tf.train.get_or_create_global_step() % 3, 0),
                             optimize_discriminator, optimize_generator)
        with tf.control_dependencies([train_step]):
            train_step = tf.train.get_global_step().assign_add(1)

    return DiscoGan(train_step=train_step, realA=A, realB=B, fakeA=fA, fakeB=fB, file_name_A=file_A, file_name_B=file_B)


DeviceMapping = namedtuple("DeviceMapping", ("input", "genA", "genB", "disA", "disB"))
