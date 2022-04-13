import tensorflow as tf
import math
import functools
from typing import Optional


def label_parser(example, mode="multiclass", num_classes=527):
    label = tf.sparse.to_dense(example['label'])
    # this just works for both cases
    # for single class, reduce sum will give 1s only on one index
    example['label'] = tf.reduce_sum(tf.one_hot(label, num_classes, on_value=1., axis=-1), axis=0)
    return example


def _decode_and_random_crop(image_bytes: tf.Tensor) -> tf.Tensor:
  """
  Make a random crop of 224.

  Copied from BYOL implementation
  https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py
  """
  img_size = tf.image.extract_jpeg_shape(image_bytes)
  area = tf.cast(img_size[1] * img_size[0], tf.float32)
  target_area = tf.random.uniform([], 0.08, 1.0, dtype=tf.float32) * area

  log_ratio = (tf.math.log(3 / 4), tf.math.log(4 / 3))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([], *log_ratio, dtype=tf.float32))

  w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
  h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

  w = tf.minimum(w, img_size[1])
  h = tf.minimum(h, img_size[0])

  offset_w = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[1] - w + 1,
                               dtype=tf.int32)
  offset_h = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[0] - h + 1,
                               dtype=tf.int32)

  crop_window = tf.stack([offset_h, offset_w, h, w])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    crop_size = (224, 224),
    jpeg_shape: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Crops to center of image with padding then scales.

    Taken from
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py
    """
    if jpeg_shape is None:
        jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = jpeg_shape[0]
    image_width = jpeg_shape[1]

    padded_center_crop_size = tf.cast(
      ((crop_size[0] / (crop_size[1] + 32)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
    ])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image


def map_dtype(example, desired=tf.float32):
    example['input'] = tf.cast(example['input'], desired)
    example['label'] = tf.cast(example['label'], desired)
    return example


def preprocess_image(
    image_bytes: tf.Tensor,
    crop_size=(224, 224),
    mode = "train",
) -> tf.Tensor:
    """Returns processed and resized images."""
    if mode == "train":
        image = _decode_and_random_crop(image_bytes)
        image = tf.image.random_flip_left_right(image)
    else:
        image = _decode_and_center_crop(image_bytes, crop_size=crop_size)

    # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
    # clamping overshoots. This means values returned will be outside the range
    # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, crop_size, tf.image.ResizeMethod.BICUBIC)
    image = tf.clip_by_value(image / 255., 0., 1.)

    # pytorch like pre-processing
    # tf broadcasting automatically applies these on the last axis
    # outputs confirmed with torch/tf code
    mean_rgb = tf.cast((0.485, 0.456, 0.406), tf.float32)
    stddev_rgb = tf.cast((0.229, 0.224, 0.225), tf.float32)
    image = image - mean_rgb
    image = image / stddev_rgb
    return image
