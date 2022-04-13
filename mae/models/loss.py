"""
Masked Autoencoder loss implementation

Written by / Copyright 2022, Sarthak Yadav
"""
import jax
import jax.numpy as jnp


def mae_loss(pred, target, mask, norm_pix_loss: bool = False):
    if norm_pix_loss:
        mean = target.mean(axis=-1, keepdims=True)
        var = target.var(axis=-1, keepdims=True)
        target = (target - mean) / (var + 1e-6) ** .5
    loss = (pred - target) ** 2
    loss = loss.mean(axis=-1)

    loss = (loss * mask).sum() / mask.sum()
    return loss
