import tensorflow as tf
import numpy as np
import time


def lrelu(x):
    with tf.name_scope("LReLu"):
        return tf.maximum(x, 0.2 * x)


def translation_generator(image, layers, out_shape):
    # Encoder
    hidden = image
    for layer in range(layers):
        hidden = tf.layers.conv2d(hidden, 64, 4, 1, activation=lrelu, padding="SAME")

    # Decoder
    for layer in range(layers-1):
        hidden = tf.layers.conv2d_transpose(hidden, 64, 4, 1, activation=tf.nn.relu, padding="SAME")

    new_image = tf.layers.conv2d_transpose(hidden, 3, 4, 1, activation=tf.nn.tanh, padding="SAME")
    return new_image


def discriminator(image, layers):
    hidden = image
    for layer in range(layers):
        hidden = tf.layers.conv2d(hidden, 32 * 2**layer, 4, 2, activation=lrelu)

    shape = hidden.shape
    total = np.prod(shape.as_list()[1:])
    flat = tf.reshape(hidden, (-1, total))
    final = tf.layers.dense(flat, 1)
    return final


def disco_gan(A, B, generator, discriminator):
    tf.summary.image("A", A)
    tf.summary.image("B", B)

    with tf.variable_scope("genA"):
        fA = generator(B)
        tf.summary.image("fA", fA)

    with tf.variable_scope("genB"):
        fB = generator(A)
        tf.summary.image("fB", fB)

    # reconstruction
    with tf.variable_scope("genA", reuse=True):
        rA = generator(fB)
        tf.summary.image("rA", rA)

    with tf.variable_scope("genB", reuse=True):
        rB = generator(fA)
        tf.summary.image("rB", rB)

    # discriminators
    with tf.variable_scope("disA"):
        dfA = discriminator(fA)

    with tf.variable_scope("disA", reuse=True):
        drA = discriminator(A)

    with tf.variable_scope("disB"):
        dfB = discriminator(fB)

    with tf.variable_scope("disB", reuse=True):
        drB = discriminator(B)

    # now all the loss terms
    dfA_l = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dfA), logits=dfA)
    dfA_lg = tf.losses.sigmoid_cross_entropy(tf.ones_like(dfA), dfA)
    drA_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drA), drA)

    dfB_l = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dfB), dfB)
    dfB_lg = tf.losses.sigmoid_cross_entropy(tf.ones_like(dfB), dfB)
    drB_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(drB), drB)

    dcA_l = tf.losses.mean_squared_error(A, rA, reduction=tf.losses.Reduction.MEAN)
    dcB_l= tf.losses.mean_squared_error(B, rB, reduction=tf.losses.Reduction.MEAN)

    loss_AB = dfB_lg + dcB_l
    loss_BA = dfA_lg + dcA_l

    loss_dA = drA_l + dfA_l
    loss_dB = drB_l + dfB_l

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

    with tf.device("/cpu:0"):
        opt_AB = optimizer_GAB.minimize(loss_AB, var_list=var_GAB)
        opt_BA = optimizer_GBA.minimize(loss_BA, var_list=var_GBA)
        opt_DA = optimizer_DA.minimize(loss_dA, var_list=var_DA)
        opt_DB = optimizer_DB.minimize(loss_dB, var_list=var_DB, global_step=tf.train.get_or_create_global_step())

    train_step = tf.group(opt_AB, opt_BA, opt_DA, opt_DB)

    return train_step


def preprocess_images(image, crop_size=108, image_size=64):
    # cropping and resizing
    cropped = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    resized = tf.image.resize_images(cropped, [image_size, image_size])
    resized.set_shape([image_size, image_size, 3])

    # transform to float in [-1, 1]
    resized = (tf.cast(resized, tf.float32) / 255.0) * 2.0 - 1.0
    return resized


def input_from_files(pattern):
    def make_input():
        with tf.variable_scope("input_fn"):
            filenames = tf.train.match_filenames_once(pattern)
            filenames = tf.train.string_input_producer(filenames, 100, shuffle=False, capacity=10)
            reader = tf.WholeFileReader()
            file_name, image_file = reader.read(filenames)
            image = tf.image.decode_jpeg(image_file, name="decode", channels=3)
            #image = tf.Print(image, data=[file_name])

        return {"image": image, "file_name": file_name, "shape": tf.shape(image)}, None
    return make_input


def input_fn(pattern):
    with tf.device("/cpu:0"):
        ip = input_from_files(pattern)
        def input_fn():
            images = ip()[0]
            images["image"] = preprocess_images(images["image"], 256)
            return tf.train.shuffle_batch([images["image"]], batch_size=32, capacity=200, min_after_dequeue=10, num_threads=2)
        return input_fn


generator = lambda x: translation_generator(x, 3, (64, 64))
disc_fn = lambda x: discriminator(x, 3)

p1 = "trainA/*.jpg"
p2 = "trainB/*.jpg"

with tf.Graph().as_default():

    train_disco = disco_gan(input_fn(p1)(), input_fn(p2)(), generator, disc_fn)

    with tf.train.MonitoredTrainingSession(checkpoint_dir="test", save_summaries_steps=1,
                                           config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        while True:
            t = time.time()
            sess.run(train_disco)
            print(time.time() - t)