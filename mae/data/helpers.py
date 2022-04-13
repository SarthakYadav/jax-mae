import functools

import pandas as pd

import tensorflow as tf
from functools import partial
import ml_collections
from . import dataset
from . import transforms


def prepare_datasets(config: ml_collections.ConfigDict, batch_size, input_dtype=tf.float32):
    train_files = pd.read_csv(config.data.tr_manifest)['files'].values
    val_files = pd.read_csv(config.data.eval_manifest)['files'].values
    label_parser_func = partial(transforms.label_parser, mode=config.model.type,
                                num_classes=config.model.num_classes)

    image_size = config.input_shape[:2]

    tr_img_parser = functools.partial(transforms.preprocess_image, crop_size=image_size, mode="train")
    val_img_parser = functools.partial(transforms.preprocess_image, crop_size=image_size, mode="eval")

    parse_record_train = functools.partial(dataset.parse_tfrecord, 
                                           image_parser=tr_img_parser,
                                           label_parser=label_parser_func,
                                           image_transforms=[])
    parse_record_val = functools.partial(dataset.parse_tfrecord, 
                                         image_parser=val_img_parser,
                                         label_parser=label_parser_func,
                                         image_transforms=[])

    train_dataset = dataset.get_dataset(train_files, batch_size, parse_example=parse_record_train,
                                        compression=config.data.get("compression", "ZLIB"))
    val_dataset = dataset.get_dataset(val_files, batch_size, parse_example=parse_record_val,
                                      compression=config.data.get("compression", "ZLIB"))

    dtype_map_func = functools.partial(transforms.map_dtype, desired=input_dtype)
    train_dataset = train_dataset.map(dtype_map_func)
    val_dataset = val_dataset.map(dtype_map_func)

    return train_dataset, val_dataset
