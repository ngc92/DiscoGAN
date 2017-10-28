import tensorflow as tf


def convert_image(image):
    """ transform to float in [-1, 1] """
    return (tf.cast(image, tf.float32) / 255.0) * 2.0 - 1.0


def make_preprocessor(crop_size=108, image_size=64):
    def preprocess_images(image):
        # cropping and resizing
        cropped = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        resized = tf.image.resize_images(cropped, [image_size, image_size])
        resized = image
        resized.set_shape([image_size, image_size, 3])

        # transform to float in [-1, 1]
        return convert_image(resized)
    return preprocess_images


def read_image_files(pattern, repeat=1):
    file_names = tf.train.match_filenames_once(pattern)
    file_names = tf.train.string_input_producer(file_names, repeat, shuffle=False, capacity=10)
    reader = tf.WholeFileReader()
    file_name, image_file = reader.read(file_names)
    image = tf.image.decode_image(image_file, name="decode", channels=3)
    return file_name, image


def input_fn(pattern, preprocessing, batch_size=32, num_threads=2, epochs=100):
    def input_fn():
        with tf.variable_scope("input_fn"):
            file_name, image = read_image_files(pattern, epochs)

            image = preprocessing(image)
            return tf.train.shuffle_batch([image], batch_size=batch_size, capacity=200, min_after_dequeue=10,
                                          num_threads=num_threads)
    return input_fn


def tf_records_input_fn(pattern, preprocessing, batch_size=32, num_threads=2):
    def input_fn():
        with tf.variable_scope("input_fn"):
            reader = tf.TFRecordReader()
            filename_queue = tf.train.string_input_producer([pattern], num_epochs=100)
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'shape': tf.FixedLenFeature((3,), tf.int64),
                })
            image = tf.image.decode_jpeg(features["image"])

            image = preprocessing(image)
            return tf.train.shuffle_batch([image], batch_size=batch_size, capacity=200, min_after_dequeue=10,
                                          num_threads=num_threads)
    return input_fn

