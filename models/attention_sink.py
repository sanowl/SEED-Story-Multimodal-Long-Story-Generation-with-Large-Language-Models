import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

class MultimodalAttentionSink(nn.Module):
    max_length: int
    sink_size: int
    hidden_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.1
    use_learnable_sink: bool = True
    use_cross_attention: bool = True

    def setup(self):
        self.sink_tokens = self.param('sink_tokens', 
                                      nn.initializers.normal(stddev=0.02), 
                                      (self.sink_size, self.hidden_dim))
        
        self.layer_norm = nn.LayerNorm()
        self.self_attention = nn.SelfAttention(num_heads=self.num_heads, 
                                               dropout_rate=self.dropout_rate)
        
        if self.use_cross_attention:
            self.cross_attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, 
                                                                   dropout_rate=self.dropout_rate)
        
        self.ffn = nn.Sequential([
            nn.Dense(self.hidden_dim * 4),
            nn.gelu,
            nn.Dense(self.hidden_dim),
            nn.Dropout(rate=self.dropout_rate)
        ])

    def __call__(self, x, sink_tokens: Optional[jnp.ndarray] = None, training: bool = False):
        if self.use_learnable_sink:
            sink = jnp.tile(self.sink_tokens[None, :, :], (x.shape[0], 1, 1))
        else:
            assert sink_tokens is not None, "Sink tokens must be provided if use_learnable_sink is False"
            sink = sink_tokens

        # Self-attention on sink tokens
        normed_sink = self.layer_norm(sink)
        attn_sink = self.self_attention(normed_sink, deterministic=not training)
        sink = sink + attn_sink

        if self.use_cross_attention:
            # Cross-attention between sink and input
            normed_x = self.layer_norm(x)
            cross_attn = self.cross_attention(queries=sink, keys=normed_x, values=normed_x, deterministic=not training)
            sink = sink + cross_attn

        # Feed-forward network
        normed_sink = self.layer_norm(sink)
        ffn_output = self.ffn(normed_sink)
        sink = sink + ffn_output

        # Concatenate sink with the last part of the input
        return jnp.concatenate([sink, x[:, -(self.max_length - self.sink_size):]], axis=1)

    @nn.compact
    def attend_to_sink(self, x, training: bool = False):
        # This method allows external attention to the sink tokens
        sink = jnp.tile(self.sink_tokens[None, :, :], (x.shape[0], 1, 1))
        normed_x = self.layer_norm(x)
        normed_sink = self.layer_norm(sink)
        
        attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, 
                                                    dropout_rate=self.dropout_rate)
        
        attended_sink = attention(queries=normed_x, keys=normed_sink, values=normed_sink, deterministic=not training)
        return x + attended_sink