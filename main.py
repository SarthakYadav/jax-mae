# Written by / Copyright 2022, Sarthak Yadav
import os
from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
from mae import train_mae
from mae import train_supervised


FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('mode', "pretrain", 'Mode (Default: ssl, Options: [ssl, train, eval])')
flags.DEFINE_bool("no_wandb", False, "To switch off wandb_logging")
flags.DEFINE_string("pretrained_dir", None, "Directory of the pretrained SSL model")
flags.DEFINE_integer("seed", 0, "seed")
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                           f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                         FLAGS.workdir, 'workdir')

    if FLAGS.mode == "pretrain":
        train_mae.train(FLAGS.config, FLAGS.workdir, FLAGS.no_wandb, FLAGS.seed)
    elif FLAGS.mode == "train":
        if FLAGS.pretrained_dir is not None and os.path.exists(FLAGS.pretrained_dir):
            existing_pretrained_dir = FLAGS.config.model.get("pretrained", None)
            if existing_pretrained_dir is not None:
                logging.info("Overriding pretrained dir {} to {}".format(existing_pretrained_dir, FLAGS.pretrained_dir))
            FLAGS.config.model.pretrained = FLAGS.pretrained_dir
        train_supervised.train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.no_wandb, FLAGS.seed)
    else:
        raise ValueError(f"Unsupported FLAGS.training_mode: {FLAGS.mode}. Supported are ['pretrain', 'train', 'eval']")


if __name__ == '__main__':
    flags.mark_flags_as_required(['workdir'])
    app.run(main)
