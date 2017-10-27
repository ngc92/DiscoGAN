import tensorflow as tf


def make_preprocessor(crop_size=108, image_size=64):
    def preprocess_images(image):
        # cropping and resizing
        cropped = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        resized = tf.image.resize_images(cropped, [image_size, image_size])
        resized.set_shape([image_size, image_size, 3])

        # transform to float in [-1, 1]
        resized = (tf.cast(resized, tf.float32) / 255.0) * 2.0 - 1.0
        return resized
    return preprocess_images


def input_fn(pattern, preprocessing, batch_size=32, num_threads=2):
    def input_fn():
        with tf.variable_scope("input_fn"):
            file_names = tf.train.match_filenames_once(pattern)
            file_names = tf.Print(file_names, [file_names], message="Using the following files")
            file_names = tf.train.string_input_producer(file_names, 100, shuffle=False, capacity=10)
            reader = tf.WholeFileReader()
            file_name, image_file = reader.read(file_names)
            image = tf.image.decode_image(image_file, name="decode", channels=3)

        image = preprocessing(image)
        return tf.train.shuffle_batch([image], batch_size=batch_size, capacity=200, min_after_dequeue=10, num_threads=num_threads)
    return input_fn

