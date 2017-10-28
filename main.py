import time
from collections import namedtuple
import tensorflow as tf

from disco.architectures import make_translation_generator, make_discriminator
from input import input_fn, make_preprocessor


DeviceMapping = namedtuple("DeviceMapping", ("input", "genA", "genB", "disA", "disB"))


def disco_gan(input_A, input_B, generator, discriminator, device_mapping):
    # create and summarize the inputs
    with tf.device(device_mapping.input):
        A = input_A()
        B = input_B()
        tf.summary.image("A", A)
        tf.summary.image("B", B)

    # create and summarize fakes
    with tf.device(device_mapping.genA), tf.variable_scope("genA"):
        fA = generator(B)
        tf.summary.image("fA", fA)

    with tf.device(device_mapping.genB), tf.variable_scope("genB"):
        fB = generator(A)
        tf.summary.image("fB", fB)

    # reconstruction
    with tf.device(device_mapping.genA), tf.variable_scope("genA", reuse=True):
        rA = generator(fB)
        tf.summary.image("rA", rA)

    with tf.device(device_mapping.genB), tf.variable_scope("genB", reuse=True):
        rB = generator(fA)
        tf.summary.image("rB", rB)

    # discriminators
    with tf.device(device_mapping.disA), tf.variable_scope("disA"):
        dfA = discriminator(fA)

    with tf.device(device_mapping.disA), tf.variable_scope("disA", reuse=True):
        drA = discriminator(A)

    with tf.device(device_mapping.disB), tf.variable_scope("disB"):
        dfB = discriminator(fB)

    with tf.device(device_mapping.disB), tf.variable_scope("disB", reuse=True):
        drB = discriminator(B)

    # now all the loss terms
    with tf.name_scope("DA_loss"):
        dfA_l = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dfA), logits=dfA, scope="fake_loss")
        drA_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drA), drA, scope="real_loss")
        loss_dA = drA_l + dfA_l

    with tf.name_scope("DB_loss"):
        dfB_l = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dfB), dfB, scope="fake_loss")
        drB_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drB), drB, scope="real_loss")
        loss_dB = drB_l + dfB_l

    with tf.name_scope("GAB_loss"):
        dfB_lg = tf.losses.sigmoid_cross_entropy(tf.ones_like(dfB), dfB, scope="discrimination_loss")
        dcB_l= tf.losses.mean_squared_error(B, rB, reduction=tf.losses.Reduction.MEAN, scope="reconstruction_loss")
        loss_AB = dfB_lg + dcB_l

    with tf.name_scope("GBA_loss"):
        dcA_l = tf.losses.mean_squared_error(A, rA, reduction=tf.losses.Reduction.MEAN, scope="reconstruction_loss")
        dfA_lg = tf.losses.sigmoid_cross_entropy(tf.ones_like(dfA), dfA, scope="discrimination_loss")
        loss_BA = dfA_lg + dcA_l

    tf.summary.scalar("loss_AB", loss_AB)
    tf.summary.scalar("loss_BA", loss_BA)
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
    optimizer_DA = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    optimizer_DB = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)

    opt_AB = optimizer_GAB.minimize(loss_AB, var_list=var_GAB, colocate_gradients_with_ops=True)
    opt_BA = optimizer_GBA.minimize(loss_BA, var_list=var_GBA, colocate_gradients_with_ops=True)
    opt_DA = optimizer_DA.minimize(loss_dA, var_list=var_DA, colocate_gradients_with_ops=True)
    opt_DB = optimizer_DB.minimize(loss_dB, var_list=var_DB, global_step=tf.train.get_or_create_global_step(),
                                   colocate_gradients_with_ops=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.group(opt_AB, opt_BA, opt_DA, opt_DB)

    return train_step


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