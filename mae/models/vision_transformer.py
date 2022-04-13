"""
Vision Transformer (ViT) in Jax

Implementation of the paper
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

Official code in Jax: https://github.com/google-research/vision_transformer

Acknowledgements/Comments:
- Wrote this to be as close to the implementation used by the official MAE implementation in
  https://github.com/facebookresearch/mae
- Which means, it's based on PyTorch implementation by Ross Wightman
  (https://github.com/rwightman/pytorch-image-models/blob/v0.3.3/timm/models/vision_transformer.py)

Written in flax / Copyright 2022, Sarthak Yadav
"""
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Union, Callable, Optional
from .drop import DropPath
from .mlp import Mlp
from .utils import constant_init
from .patch_embed import PatchEmbed

dense_kernel_init = nn.initializers.xavier_uniform()


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.

    @nn.compact
    def __call__(self, x, train: bool = True):
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        qkv_layer = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, kernel_init=dense_kernel_init)
        proj_layer = nn.Dense(self.dim, kernel_init=dense_kernel_init)

        B, N, C = x.shape
        qkv = qkv_layer(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv
        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = nn.softmax(attn, axis=-1)
        if self.attn_drop != 0:
            attn = nn.Dropout(self.attn_drop, deterministic=not train, name="attn_drop_layer")(attn)
        x = jnp.swapaxes((attn @ v), 1, 2).reshape(B, N, C)
        x = proj_layer(x)
        if self.proj_drop != 0:
            x = nn.Dropout(self.proj_drop, deterministic=not train, name="proj_drop_layer")(x)
        return x


class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        gamma = self.param("gamma", partial(constant_init, constant=self.init_values), [self.dim])
        return x * gamma


class Block(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    drop: float = 0.
    attn_drop: float = 0.
    init_values: Any = None
    drop_path: float = 0.
    act_layer: Union[Callable, nn.Module] = nn.gelu
    norm_layer: Union[Callable, nn.Module] = nn.LayerNorm

    @nn.compact
    def __call__(self, x, train: bool = True):
        outputs1 = self.norm_layer()(x)
        outputs1 = Attention(self.dim, num_heads=self.num_heads,
                             qkv_bias=self.qkv_bias, attn_drop=self.attn_drop,
                             proj_drop=self.drop)(outputs1, train=train)

        if self.init_values is not None:
            outputs1 = LayerScale(self.dim, init_values=self.init_values)(outputs1)

        if self.drop_path > 0.:
            outputs1 = DropPath(self.drop_path)(outputs1, train=train)

        x = x + outputs1

        outputs2 = self.norm_layer()(x)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        outputs2 = Mlp(hidden_features=mlp_hidden_dim, out_features=self.dim,
                       drop=self.drop, kernel_init=dense_kernel_init,
                       activation=self.act_layer)(outputs2, train=train)

        if self.init_values is not None:
            outputs2 = LayerScale(self.dim, init_values=self.init_values)(outputs2)

        if self.drop_path > 0.:
            outputs2 = DropPath(self.drop_path)(outputs2, train=train)
        x = x + outputs2
        return x


class BNWrapper(nn.Module):
    use_running_average: bool = True
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.BatchNorm(use_running_average=not train, use_bias=self.use_bias, 
                             use_scale=self.use_scale, name='head_norm')(x)
        return x


class VisionTransformer(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.
    norm_layer: Optional[Callable] = nn.LayerNorm
    global_pool: bool = False
    lin_probe: bool = False
    dtype: Any = None

    def setup(self):
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = self.param("cls_token", nn.initializers.normal(0.02),
                                    [1, 1, self.embed_dim])

        self.pos_embed = self.param("pos_embed", nn.initializers.normal(0.02),
                               [1, self.num_patches+1, self.embed_dim])
        # TODO: make a drop out wrapper like BNWrapper above to allow easy initializing
        # if self.drop_rate > 0.:
        #     self.pos_drop = partial(nn.Dropout, rate=self.drop_rate, name="pos_drop")
        dpr = [x for x in np.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = [
            Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                  qkv_bias=self.qkv_bias, drop=self.drop_rate,
                  attn_drop=self.attn_drop_rate, drop_path=dpr[i],
                  norm_layer=self.norm_layer, name="encoder_block_{:02d}".format(i))
            for i in range(self.depth)
        ]
        self.encoder_norm = self.norm_layer(name="encoder_norm")
        if self.global_pool:
            self.fc_norm = self.norm_layer(name="fc_norm")

        # in my initial experiments, BatchNorm gave worse results
        # This might be changed in future releases as I have more experimental data

        # if self.lin_probe:
            # BatchNorm without affine was used for lin-probe in MAE
            # self.head_norm = BNWrapper(use_bias=False, use_scale=False, name='head_norm')
            # self.head_norm = self.norm_layer(name="head_norm")
            # self.head_norm = partial(nn.BatchNorm, use_bias=False, 
            #                          use_scale=True, )

        self.head = nn.Dense(self.num_classes)

    def forward_features(self, x, train: bool = True):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = jnp.broadcast_to(self.cls_token, (x.shape[:1] + self.cls_token.shape[1:]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        if self.drop_rate > 0.:
            x = self.pos_drop(deterministic=not train)(x)
        for blk in self.blocks:
            x = blk(x, train=train)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.encoder_norm(x)
            outcome = x[:, 0]
        return outcome

    def __call__(self, x, train: bool = True):
        x = self.forward_features(x, train)
        # if self.lin_probe:
        #     x = self.head_norm(x)
        x = self.head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model
