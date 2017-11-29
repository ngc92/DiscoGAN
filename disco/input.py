import tensorflow as tf


class Pipe:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __or__(self, g):
        f = self.f

        def chained(*args, **kwargs):
            return g(f(*args, **kwargs))
        return Pipe(chained)


def convert_image():
    """ transform to float in [-1, 1] """
    def f(image):
        return (tf.cast(image, tf.float32) / 255.0) * 2.0 - 1.0
    return Pipe(f)


def crop_and_resize_image(crop_size, image_size):
    def f(image):
        # cropping and resizing
        if crop_size == "min":
            cropping = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
        elif crop_size == "max":
            cropping = tf.maximum(tf.shape(image)[0], tf.shape(image)[1])
        else:
            cropping = crop_size

        with tf.name_scope("crop_or_pad"):
            cropped = tf.image.resize_image_with_crop_or_pad(image, cropping, cropping)
        tf.summary.image("cropped", cropped[None, :, :, :])
        with tf.name_scope("resize"):
            resized = tf.image.resize_images(cropped, [image_size, image_size])
            resized.set_shape([image_size, image_size, 3])
        tf.summary.image("resized", resized[None, :, :, :])
        return tf.cast(resized, tf.uint8)
    return Pipe(f)


def thicken():
    def f(image):
        return tf.layers.max_pooling2d(tf.expand_dims(image, axis=0), 2, 1, "same")[0, :, :, :]
    return Pipe(f)


def random_crop(crop_size, image_size=None, seed=None):
    def f(image):
        cropped = tf.random_crop(image, [crop_size, crop_size, 3], seed=seed)
        if image_size is not None:
            resized = tf.image.resize_images(cropped, [image_size, image_size])
            resized.set_shape([image_size, image_size, 3])
        else:
            resized = cropped
        return tf.cast(resized, tf.uint8)
    return Pipe(f)


def augment_with_flips(horizontal=True, vertical=False, seed=None):
    def f(image):
        with tf.name_scope("augment_with_flips", [image]):
            if horizontal:
                image = tf.image.random_flip_left_right(image, seed=seed)
            if vertical:
                image = tf.image.random_flip_up_down(image, seed=None if seed is None else seed+1)
            return image
    return Pipe(f)


def augment_with_rotations(seed=None):
    def f(image):
        with tf.name_scope("augment_with_rotation", [image]):
            rotate = tf.random_uniform((), 0, 4, tf.int32, seed=seed)
            return tf.image.rot90(image, rotate)

    return Pipe(f)


def read_image_files(pattern, repeat=1):
    file_names = tf.train.match_filenames_once(pattern)
    file_names = tf.train.string_input_producer(file_names, repeat, shuffle=False, capacity=10)
    reader = tf.WholeFileReader()
    file_name, image_file = reader.read(file_names)
    image = tf.image.decode_image(image_file, name="decode", channels=3)
    image.set_shape([None, None, 3])
    return file_name, image


def input_pipeline(pattern, preprocessing, batch_size=32, num_threads=2, epochs=100):
    def input_fn():
        with tf.variable_scope("input_fn"):
            file_name, image = read_image_files(pattern, epochs)

            image = preprocessing(image)
            return tf.train.shuffle_batch([image, file_name], batch_size=batch_size, capacity=200, min_after_dequeue=10,
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

