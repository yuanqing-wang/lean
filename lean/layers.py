import jax
import jax.numpy as jnp
from flax import linen as nn
EPSILON = 1e-10

class EGNNLayer(nn.Module):
    hidden_features: int
    out_features: int
    
    @nn.compact
    def __call__(self, h, x):
        delta_x = x[..., :, None, :] - x[..., None, :, :]
        distance = ((delta_x ** 2).sum(-1, keepdims=True) + EPSILON )** 0.5
        message = nn.Dense(self.hidden_features)(
            jnp.concatenate(
                [
                    h[..., :, None, :].repeat(h.shape[-2], axis=-2),
                    h[..., None, :, :].repeat(h.shape[-2], axis=-3),
                    distance,
                ],
                axis=-1,
            )
        )
        x = x + (delta_x * jax.nn.tanh(nn.Dense(1)(message))).mean(-2)
        aggregated_message = message.mean(-2)
        h = nn.Dense(self.out_features)(
            jnp.concatenate([h, aggregated_message], axis=-1),
        )
        return h, x