"""
Implementation of MaskedAutoencoder from the paper

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár & Ross Girshick,
"Masked Autoencoders Are Scalable Vision Learners", https://arxiv.org/abs/2111.06377, 2021

Adapted from the official PyTorch implementation here: https://github.com/facebookresearch/mae

Adapted to Jax/modifications / Copyright 2022, Sarthak Yadav
"""
import jax
from functools import partial
import jax.numpy as jnp
import flax.linen as nn
from .patch_embed import PatchEmbed
from .vision_transformer import Block
from .pos_embed import get_2d_sincos_pos_embed
from .utils import constant_init
from typing import Optional, Callable, Any


def unbatched_gather(x, ids_keep):
    return x[ids_keep, Ellipsis]


batched_gather = jax.vmap(unbatched_gather)


class MaskedAutoencoderViT(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.
    norm_layer: Optional[Callable] = nn.LayerNorm
    dtype: Any = jnp.float32

    def setup(self):
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = self.param("cls_token", nn.initializers.normal(0.02),
                               [1, 1, self.embed_dim])

        # this is a variable (and not a param), and is not learned
        # but it's placed in the "params" dictionary for easier finetuning
        self.pos_embed = self.variable("params", "pos_embed",
                                       init_fn=partial(get_2d_sincos_pos_embed,
                                                       embed_dim=self.embed_dim,
                                                       grid_size=int(self.patch_embed.num_patches**.5),
                                                       cls_token=True, expand_first_dim=True),
                                       )

        self.blocks = [Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer, name="encoder_block_{:02d}".format(i)) for i in range(self.depth)]

        self.encoder_norm = self.norm_layer(name="encoder_norm")

        self.decoder_embed = nn.Dense(self.decoder_embed_dim, use_bias=True)
        self.mask_token = self.param("mask_token", nn.initializers.normal(0.02),
                                     [1, 1, self.decoder_embed_dim])

        self.decoder_pos_embed = self.variable("params", "decoder_pos_embed",
                                               init_fn=partial(get_2d_sincos_pos_embed,
                                                               embed_dim=self.decoder_embed_dim,
                                                               grid_size=int(self.patch_embed.num_patches ** .5),
                                                               cls_token=True, expand_first_dim=True))

        self.decoder_blocks = [Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True,
                                     norm_layer=self.norm_layer, name="decoder_block_{:02d}".format(i))
                               for i in range(self.decoder_depth)]
        self.decoder_pred = nn.Dense(self.patch_size**2 * self.in_chans, use_bias=True)
        self.decoder_norm = self.norm_layer(name="decoder_norm")

        rng = self.make_rng("random_masking")

    def patchify(self, imgs):
        """
        imgs: (N, H, W, 3)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape((imgs.shape[0], h, p, w, p, 3))
        x = jnp.einsum('nhpwqc->nhwpqc', x)
        x = x.reshape((imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, H, W, 3)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, 3))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, h * p, 3))
        return imgs

    def random_masking(self, x, mask_ratio, rng):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = jax.random.uniform(rng, (N, L))

        # sort noise for each sample
        ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = batched_gather(x, ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones((N, L))
        mask = mask.at[:, :len_keep].set(0)

        mask = batched_gather(mask, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, train: bool = True, rng=None):
        x = self.patch_embed(x)
        # x = x + self.pos_embed[:, 1:, :]
        # print(self.pos_embed.value.shape)
        pos_embed = self.pos_embed.value
        x = x + pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, rng)
        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = jnp.broadcast_to(cls_token, (x.shape[:1] + cls_token.shape[1:]))

        x = jnp.concatenate([cls_tokens, x], axis=1)
        for blk in self.blocks:
            x = blk(x, train=train)

        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, train: bool = True):
        x = self.decoder_embed(x)
        # mask_token = self.param("mask_token", nn.initializers.normal(0.02),
        #                         [1, 1, self.decoder_embed_dim])
        mask_tokens = jnp.broadcast_to(self.mask_token,
                                       (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], x.shape[-1]))
        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)
        x_ = batched_gather(x_, ids_restore)
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)
        decoder_pos_embed = self.decoder_pos_embed.value
        x = x + decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x, train=train)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def __call__(self, imgs, mask_ratio: float = 0.75, train: bool = True, rng=None):
        if rng is None:
            rng = self.make_rng("random_masking")
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, train=train, rng=rng)
        pred = self.forward_decoder(latent, ids_restore, train=train)
        target = self.patchify(imgs)

        return pred, target, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
