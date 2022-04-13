"""
Dataset and stuff. Although tfds could be used, I had my own imagenet tfrecords.
Integration with tfds should be trivial

Written in jax/Copyright 2022, Sarthak Yadav
"""
import os
import tensorflow as tf
from . import transforms

GCS_PATH = os.environ.get('GCS_PATH', None)


def parse_tfrecord(example, image_parser, 
                   label_parser=None, image_transforms=[]):
    feature_description = {
        "input": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    # example['input'] = tf.io.decode_jpeg(example['input'])
    image = example['input']
    image = image_parser(image)
    image_shape = tf.shape(image)[-1]
    if image_shape != 3:
        image = tf.repeat(image, repeats=[3], axis=-1)
    for tfs in image_transforms:
        image = tfs(image)
    example['input'] = image
    if label_parser:
        example = label_parser(example)
    return example


def get_dataset(filenames, batch_size,
                parse_example,
                compression="ZLIB",
                cacheable=False):
    options = tf.data.Options()
    options.autotune.enabled = True
    options.threading.private_threadpool_size = 96  # 0=automatically determined
    options.deterministic = False
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(options)
    dataset = dataset.shuffle(len(filenames), seed=0, reshuffle_each_iteration=True)
    if cacheable:
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=compression,
                                              num_parallel_reads=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE),
            cycle_length=tf.data.AUTOTUNE, block_length=32,
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=compression,
                                              num_parallel_reads=tf.data.AUTOTUNE).map(parse_example,
                                                                                       num_parallel_calls=tf.data.AUTOTUNE),
            cycle_length=tf.data.AUTOTUNE, block_length=32,
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
