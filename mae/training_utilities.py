"""
Helper functions for jax/flax training

Written by / Copyright 2022, Sarthak Yadav
"""
import os
import json
import logging
import functools
import ml_collections
import optax
from typing import Any
import jax
from jax import lax, random
import flax
from flax import jax_utils
from jax import numpy as jnp
from flax.training import checkpoints
from flax.training import train_state
from flax.training import common_utils
from flax import optim
# from .misc import TrainingMode, Features, DataSplit
from flax.core import freeze, unfreeze
from .trainstate import TrainState_v2
from sklearn.metrics import average_precision_score
import enum
from . import models
from flax.training import common_utils


@enum.unique
class TrainingMode(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    CONTRASTIVE = "contrastive"
    MAE = "mae"
    COLA = "cola"


def get_dtype(half_precision: bool):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_dtype


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def save_best_checkpoint(state, workdir, best_acc):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, prefix="best_ckpt_", keep=3)


def initialize(key, inp_shape, model,
               aux_rng_keys=["dropout", "drop_path", "mixup", "spec_aug", "random_masking"]):
    input_shape = (2,) + inp_shape

    @jax.jit
    def init(*args):
        return model.init(*args)

    num_keys = len(aux_rng_keys)
    key, *subkeys = jax.random.split(key, num_keys + 1)
    rng_keys = {aux_rng_keys[ix]: subkeys[ix] for ix in range(len(aux_rng_keys))}
    variables = init({'params': key, **rng_keys},
                     jnp.ones(input_shape, model.dtype))
    rngs = flax.core.FrozenDict(rng_keys)
    if "batch_stats" in variables.keys():
        return variables['params'], variables['batch_stats'], rngs
    else:
        return variables['params'], flax.core.freeze({}), rngs


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    if len(state.batch_stats) != 0:
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    else:
        return state


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int,
        num_epochs: int = None):
    """Create learning rate schedule."""
    if config.opt.schedule == "cosine":
        logging.info("Using cosine learning rate decay")
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=base_learning_rate,
            transition_steps=config.opt.warmup_epochs * steps_per_epoch)
        if num_epochs is None:
            num_epochs = config.num_epochs
        cosine_epochs = max(num_epochs - config.opt.warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[config.opt.warmup_epochs * steps_per_epoch])
    else:
        schedule_fn = base_learning_rate
    return schedule_fn


def prepare_tf_data(xs, devices=None):
    """Convert a input batch from tf Tensors to numpy arrays."""
    if devices is None:
        local_device_count = jax.local_device_count()
    elif type(devices) == list or type(devices) == tuple:
        local_device_count = len(devices)
    else:
        raise ValueError("Devices should either be None or a list of jax.device")

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, channel) to
        # (local_devices, device_batch_size, height, width, channel)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, learning_rate_fn, **kwargs):
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = optim.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats, rng_keys = initialize(rng, config.input_shape, model, **kwargs)
    optimizer_name = config.opt.optimizer.lower()
    if optimizer_name == "adamw":
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "lars":
        tx = optax.lars(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.opt.get("momentum", 0.9),
            nesterov=config.opt.get("nesterov", False)
        )
    else:
        raise ValueError("optimizer {} not supported. Valid values are [adamw, lars, sgd]")
    grad_accum_steps = config.opt.get("grad_accum_steps", 1)
    if grad_accum_steps > 1:
        logging.info("Using gradient accumulation of {} steps".format(grad_accum_steps))
        tx = optax.MultiSteps(tx, grad_accum_steps)
    logging.info("Using {} optimizer".format(optimizer_name))
    state = TrainState_v2.create(
        apply_fn=model.apply,
        params=params,
        frozen_params=flax.core.freeze({}),
        tx=tx,
        batch_stats=batch_stats,
        aux_rng_keys=rng_keys,
        dynamic_scale=dynamic_scale)
    return state


def create_train_state_from_pretrained(rng, config: ml_collections.ConfigDict,
                                       model, learning_rate_fn,
                                       pretrained_work_dir,
                                       pretrained_prefix="checkpoint_",
                                       copy_mode="all",
                                       to_copy=[],
                                       fc_only=False):
    logging.info("Making train state from pretrained..")
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = optim.DynamicScale()
    else:
        dynamic_scale = None
    params, batch_stats, rng_keys = initialize(rng, config.input_shape, model)

    # load pretrained ckpt to a dictionary
    pretrained_state_dict = checkpoints.restore_checkpoint(pretrained_work_dir, None,
                                                           prefix=pretrained_prefix)
    pretrained_params = pretrained_state_dict['params']
    pretrained_batch_stats = pretrained_state_dict['batch_stats']

    # unfreeze classifier params and batch_stats
    params = unfreeze(params)
    batch_stats = unfreeze(batch_stats)
    if copy_mode == "all":
        to_copy = list(set(params.keys()).intersection(set(pretrained_params.keys())))
    logging.info("Copying the following parameters: \n[{}]".format(",".join(to_copy)))
    # copy stuff
    for k in to_copy:
        assert k in params.keys()  # and k in batch_stats.keys()
        params[k] = pretrained_params[k]
        try:
            batch_stats[k] = pretrained_batch_stats[k]
        except KeyError as ex:
            pass

    # filter params based on fc_only argument
    if fc_only:
        logging.info("Finetuning fc-only layer")
        frozen_params = {}
        trainable_params = {}
        for k in params.keys():
            if k in to_copy:
                frozen_params[k] = params[k]
            else:
                trainable_params[k] = params[k]
    else:
        frozen_params = {}
        trainable_params = params

    # freeze classifier params and batch_stats
    trainable_params = freeze(trainable_params)
    frozen_params = freeze(frozen_params)
    batch_stats = freeze(batch_stats)

    # make the train state now
    optimizer_name = config.opt.optimizer.lower()
    if optimizer_name == "adamw":
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "lars":
        tx = optax.lars(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.opt.get("momentum", 0.9),
            nesterov=config.opt.get("nesterov", False)
        )
    else:
        raise ValueError("optimizer {} not supported. Valid values are [adamw, lars, sgd]")
    grad_accum_steps = config.opt.get("grad_accum_steps", 1)
    if grad_accum_steps > 1:
        logging.info("Using gradient accumulation of {} steps".format(grad_accum_steps))
        tx = optax.MultiSteps(tx, grad_accum_steps)
    state = TrainState_v2.create(
        apply_fn=model.apply,
        params=trainable_params,
        frozen_params=frozen_params,
        tx=tx,
        batch_stats=batch_stats,
        aux_rng_keys=rng_keys,
        dynamic_scale=dynamic_scale)
    return state


def cross_entropy_loss(logits, labels, smoothing_factor: float = None):
    if smoothing_factor and type(smoothing_factor) == float:
        labels = optax.smooth_labels(labels, alpha=smoothing_factor)
    # print("!!!!!! cross_entropy_loss shape:", one_hot_labels.shape)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(xentropy)


def binary_xentropy_loss(logits, labels, smoothing_factor: float = None):
    if smoothing_factor and type(smoothing_factor) == float:
        labels = optax.smooth_labels(labels, alpha=smoothing_factor)
    xentropy = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels, mode, cost_fn):
    loss = cost_fn(logits, labels)
    metrics = {
        'loss': loss
    }
    if mode == TrainingMode.MULTICLASS:
        accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
        # labels is now a (batch, num_classes) onehot-encoded array
        metrics['accuracy'] = accuracy

    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def create_input_iter(ds, devices=None):
    ds = ds.repeat()
    prep_data = functools.partial(prepare_tf_data, devices=devices)
    it = map(prep_data, ds)
    it = jax_utils.prefetch_to_device(it, 10, devices=devices)
    return it


def get_eval_metrics_summary(eval_metrics, eval_logits, eval_labels, mode=TrainingMode.MULTICLASS):
    eval_metrics = common_utils.get_metrics(eval_metrics)

    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    if mode == TrainingMode.MULTICLASS:
        return summary
    else:
        # if mode == TrainingMode.MULTICLASS:
        #     logging.info('eval epoch: %d, loss: %.4f, accuracy: %.4f',
        #                  epoch, summary['loss'], summary['accuracy'] * 100)
        eval_logits = jnp.concatenate([jax.device_get(x) for x in eval_logits])
        eval_labels = jnp.concatenate([jax.device_get(x) for x in eval_labels])

        eval_logits = eval_logits.reshape(-1, eval_logits.shape[-1])
        eval_labels = eval_labels.reshape(-1, eval_labels.shape[-1])
        # print("logits.shape: {} | type: {}".format(eval_logits.shape, type(eval_logits)))

        map_value = average_precision_score(eval_labels.astype('float32'), eval_logits.astype('float32'),
                                            average="macro")

        summary['accuracy'] = map_value
        return summary
        # logging.info('eval epoch: %d, loss: %.4f, accuracy: %.4f',
        #              epoch, summary['loss'], summary['accuracy'] * 100)


def read_config_from_json(workdir):
    config_path = os.path.join(workdir, "config.json")
    print(f"loading config -> {config_path}")
    with open(config_path, "r") as fd:
        config_dict = json.load(fd)
        config_dict['input_shape'] = tuple(config_dict['input_shape'])
    config = ml_collections.ConfigDict(config_dict)
    return config


def write_config_to_json(workdir, config):
    config_path = os.path.join(workdir, "config.json")
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    if os.path.exists(config_path):
        logging.info(f"config file {config_path} exists.. Not overwriting.")
        return
    with open(config_path, "w") as fd:
        json.dump(config.to_dict(), fd)


def get_model_cls(config):
    model_cls = getattr(models, config.model.arch)
    model_args = config.model.get("model_args", None)
    if model_args:
        model_cls = functools.partial(model_cls, **model_args.to_dict())
    # if "efficient" in config.model.arch:
    #     model_cls = functools.partial(model_cls, drop_path_prob=0.)
    return model_cls


def create_mae_model(model_cls, half_precision, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(dtype=model_dtype, **kwargs)


def create_model(model_cls, half_precision, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(dtype=model_dtype, **kwargs)
