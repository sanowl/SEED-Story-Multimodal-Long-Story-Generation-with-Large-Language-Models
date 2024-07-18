import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable
from functools import partial

class GroupNorm(nn.Module):
    groups: int
    epsilon: float = 1e-5
    scale_init: Callable = nn.initializers.ones
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        return nn.GroupNorm(
            num_groups=self.groups,
            epsilon=self.epsilon,
            scale_init=self.scale_init,
            bias_init=self.bias_init
        )(x)

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5
    trainable: bool = True

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', 
                           nn.initializers.constant(self.init_values), 
                           (self.dim,),
                           trainable=self.trainable)
        return x * scale

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        return x

class QFormerLayer(nn.Module):
    config: Any
    ffn_ratio: int = 4
    use_rotary_embedding: bool = True

    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate
        )
        self.ffn = FeedForward(
            dim=self.config.query_dim,
            hidden_dim=self.config.query_dim * self.ffn_ratio,
            dropout=self.config.dropout_rate
        )
        self.norm1 = GroupNorm(self.config.groups)
        self.norm2 = GroupNorm(self.config.groups)
        self.ls1 = LayerScale(self.config.query_dim, self.config.layer_scale_init_value)
        self.ls2 = LayerScale(self.config.query_dim, self.config.layer_scale_init_value)
        
        if self.use_rotary_embedding:
            self.rotary_emb = RotaryEmbedding(self.config.query_dim // self.config.num_heads)

    def __call__(self, x, queries, mask=None, training=False):
        k, v = nn.Dense(self.config.query_dim)(x), nn.Dense(self.config.query_dim)(x)
        
        if self.use_rotary_embedding:
            queries, k = self.rotary_emb(queries), self.rotary_emb(k)
        
        attn_output = self.attention(
            self.norm1(queries), k, v,
            mask=mask,
            deterministic=not training
        )
        queries = queries + nn.Dropout(self.config.dropout_rate)(self.ls1(attn_output), deterministic=not training)
        
        ffn_output = self.ffn(self.norm2(queries), training=training)
        return queries + nn.Dropout(self.config.dropout_rate)(self.ls2(ffn_output), deterministic=not training)

class QFormer(nn.Module):
    config: Any
    use_relative_positions: bool = True
    max_relative_position: int = 32

    def setup(self):
        self.query_embed = nn.Embed(
            num_embeddings=self.config.num_queries,
            features=self.config.query_dim
        )
        self.layers = [QFormerLayer(self.config) for _ in range(self.config.num_layers)]
        
        if self.use_relative_positions:
            self.relative_position_bias = RelativePositionBias(
                num_buckets=32,
                max_distance=self.max_relative_position,
                n_heads=self.config.num_heads
            )

    def __call__(self, x, training=False):
        b, _, _ = x.shape
        queries = self.query_embed(jnp.arange(self.config.num_queries))
        queries = jnp.tile(queries[None, :, :], (b, 1, 1))
        
        mask = None
        if self.use_relative_positions:
            mask = self.relative_position_bias(self.config.num_queries, self.config.num_queries)
        
        for layer in self.layers:
            queries = layer(x, queries, mask=mask, training=training)
        
        return queries

class RotaryEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        freqs = jnp.einsum(
            "i,j->ij",
            jnp.arange(seq_len),
            1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return x * jnp.cos(emb) + jnp.roll(x, shift=1, axis=-1) * jnp.sin(emb)

class RelativePositionBias(nn.Module):
    num_buckets: int
    max_distance: int
    n_heads: int

    @nn.compact
    def __call__(self, query_length: int, key_length: int):
        context_position = jnp.arange(query_length)[:, None]
        memory_position = jnp.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        relative_position_bucket = jax.lax.stop_gradient(relative_position_bucket)
        values = self.param(
            'rel_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.n_heads, self.num_buckets)
        )
        return values[:, relative_position_bucket]

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(jnp.int32) * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.maximum(n, 0)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            jnp.log(n.astype(jnp.float32) / max_exact) /
            jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).astype(jnp.int32)
        val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
        ret += jnp.where(is_small, n, val_if_large)
        return ret