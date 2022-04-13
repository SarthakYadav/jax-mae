"""
Implementation of PatchEmbed layer

Inspired by the PyTorch implementation in timm (https://github.com/rwightman/pytorch-image-models)
by Ross Wightman, 2020

Written in jax by / Copyright 2022, Sarthak Yadav
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from .utils import to_2tuple
from typing import Optional, Callable, Union


class PatchEmbed(nn.Module):
    img_size: Optional[Union[tuple, int]] = 224
    patch_dim: Optional[Union[tuple, int]] = 16
    # in_chans: int = 3
    embed_dim: int = 768
    norm_layer: Optional[Callable] = None
    flatten: bool = True

    def setup(self):
        img_size = to_2tuple(self.img_size)
        patch_size = to_2tuple(self.patch_dim)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.proj = nn.Conv(self.embed_dim, kernel_size=patch_size, strides=patch_size, padding='VALID',
                            kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, inputs, train: bool = True):
        B, H, W, C = inputs.shape
        outputs = self.proj(inputs)
        if self.flatten:
            outputs = outputs.reshape(B, -1, self.embed_dim) # B,N,C shape
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)
        return outputs
