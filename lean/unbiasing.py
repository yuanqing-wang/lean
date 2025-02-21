import jax
from jax import numpy as jnp
from typing import NamedTuple
from functools import partial

class SinRBF(NamedTuple):
    mu_x : jnp.ndarray # (D_x, )
    mu_t : jnp.ndarray # (D_t, )
    gamma : jnp.ndarray # (D_x, D_t)
    coefficient: jnp.ndarray # (D_x, D_t)
    
    @classmethod
    def init(
        cls,
        key: jax.random.PRNGKey,
        num_x: int,
        num_t: int,
        std: float = 1e-10,
        max_distance: float = 10.0,
    ):
        gamma = jnp.ones((num_x, num_t)) # * std
        coefficient = jax.random.normal(key, (num_x, num_t)) * std
        mu_x = jnp.linspace(0, max_distance, num_x)
        mu_t = jnp.linspace(0, 1, num_t)
        return cls(mu_x, mu_t, gamma, coefficient)        

    def _call_single(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
    ):
        original_t = t
        t = t - self.mu_t
        x = x - self.mu_x
        x = x[:, None]
        t = t[None, :]
        r = x**2 + t**2
        r = jax.nn.softplus(self.gamma) * r
        r = jnp.exp(-r)
        r = (r * self.coefficient).sum()
        sin = jnp.sin(original_t * jnp.pi)
        return r * sin
    
    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
    ):
        distances = (((x[..., :, None] - x[..., None, :]) ** 2).sum(-1) + 1e-5) ** 0.5
        distances = distances.flatten()[:, None]
        energies = jax.vmap(partial(self._call_single, t=time))(distances)
        return energies.sum()
        
        
    
    
    