"""
Training a Masked Autoencoder

Written by / Copyright 2022, Sarthak Yadav
"""
import copy
import functools
import time
from typing import Any
from absl import logging
from clu import metric_writers
from clu import periodic_actions
import jax
from jax import lax, random
import jax.numpy as jnp
from . import training_utilities
import flax
from flax import jax_utils
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import ml_collections
import optax
import wandb
import tensorflow as tf
from mae.data import helpers
from mae.models import loss, mae

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def compute_metrics(loss):
    metrics = {
        'loss': loss
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def train_step(state, batch, learning_rate_fn,
               cost_func, 
               mode=training_utilities.TrainingMode.MAE):
    def loss_fn(params):
        inputs = batch['input']
        labels = batch['label']
        (pred, target, mask), new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            inputs,
            mutable=['batch_stats'],
            rngs=state.aux_rng_keys
        )
        # labels = common_utils.onehot(jnp.arange(0, logits.shape[0]), num_classes=logits.shape[0])
        loss = cost_func(pred, target, mask)
        return loss, (new_model_state, pred, labels)

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(
            loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name='batch')
    new_model_state, pred, labels = aux[1]
    loss = aux[0]
    metrics = compute_metrics(loss)
    metrics['learning_rate'] = lr
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])
    if dynamic_scale:
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params))
        metrics['scale'] = dynamic_scale.scale
    return new_state, metrics


def train(config: ml_collections.ConfigDict,
          workdir: str,
          no_wandb: bool,
          seed: int = 0):
    wandb_logger = None
    if not no_wandb:
        wandb_logger = wandb.init(project='{}'.format(config.wandb.get("project", "audax-cola")),
                                  group="{}".format(config.data.dataset_name),
                                  config=config.to_dict(), name=workdir.split("/")[-1])
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    training_utilities.write_config_to_json(workdir, config)
    rng = random.PRNGKey(seed)
    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.process_count()
    logging.info("Process count: {}".format(jax.process_count()))
    device = config.get("device", None)
    if device is not None:
        devices = [jax.local_devices()[device]]
    else:
        devices = jax.local_devices()
    print("Training on the following devices: {}".format(devices))
    platform = devices[0].platform
    if config.half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32
    mode = training_utilities.TrainingMode(config.model.type)
    assert mode == training_utilities.TrainingMode.MAE
    train_iter, eval_iter = helpers.prepare_datasets(config, local_batch_size, input_dtype=input_dtype)
    train_iter = training_utilities.create_input_iter(train_iter, devices=devices)
    eval_iter = training_utilities.create_input_iter(eval_iter, devices=devices)

    num_examples = config.data.tr_samples
    if config.get("steps_per_epoch", -1) == -1:
        steps_per_epoch = (num_examples // config.batch_size)
    else:
        steps_per_epoch = config.get("steps_per_epoch")
    
    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
        num_epochs = config.num_epochs
    else:
        num_steps = config.num_train_steps
        num_epochs = config.num_train_steps // steps_per_epoch
    logging.info("num_steps: {} | num_epochs: {}".format(num_steps, num_epochs))
    
    if config.steps_per_eval == -1:
        num_validation_examples = config.data.eval_samples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval
    
    steps_per_checkpoint = steps_per_epoch
    base_learning_rate = config.opt.get("grad_accum_steps", 1) * config.opt.learning_rate * config.batch_size / 256.

    model_cls = training_utilities.get_model_cls(config)
    if mode == training_utilities.TrainingMode.MAE:
        cost_fn = functools.partial(loss.mae_loss, norm_pix_loss=config.opt.get("norm_pix_loss", False))
        model = training_utilities.create_mae_model(model_cls, half_precision=config.half_precision)
    else:
        raise ValueError("Unsupported mode {}".format(mode))
    
    print(model)
    learning_rate_fn = training_utilities.create_learning_rate_fn(
        config, base_learning_rate, steps_per_epoch, num_epochs=num_epochs)
    logging.info('Creating train state...')
    state = training_utilities.create_train_state(rng, config, model, learning_rate_fn)
    state = training_utilities.restore_checkpoint(state, workdir)
    step_offset = int(state.step)
    state = jax_utils.replicate(state, devices=devices)
    logging.info('Train state ready...')
    p_train_step = jax.pmap(
        functools.partial(train_step, 
                          learning_rate_fn=learning_rate_fn,
                          cost_func=cost_fn,
                          mode=mode),
        axis_name='batch', devices=devices)
    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        is_best_ckpt = False
        if step == 0:
            print(batch['input'].shape, batch['input'].dtype)
        state, metrics = p_train_step(state, batch)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')
        if config.get('log_every_steps'):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f'train_{k}': v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary['steps_per_second'] = config.log_every_steps / (
                        time.time() - train_metrics_last_t)
                
                writer.write_scalars(step + 1, copy.copy(summary))
                if wandb_logger:
                    wandb_logger.log(summary, step+1)
                train_metrics = []
                train_metrics_last_t = time.time()
        
        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            state = training_utilities.sync_batch_stats(state)
            training_utilities.save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    if wandb_logger:
        wandb_logger.finish()
    return state
