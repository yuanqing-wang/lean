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
        t0 = t
        t = t - self.mu_t
        x = x - self.mu_x
        x = x[:, None]
        t = t[None, :]
        r = x**2 + t**2
        r = jax.nn.softplus(self.gamma) * r
        r = jnp.exp(-r)
        r = (r * self.coefficient).sum()
        return r # * jnp.sin(2 * jnp.pi * t0)
    
    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
    ):
        distances = (((x[..., :, None] - x[..., None, :]) ** 2).sum(-1) + 1e-5) ** 0.5
        distances = distances.flatten()[:, None]
        energies = jax.vmap(partial(self._call_single, t=time))(distances)
        return energies.sum()
        
        # return jnp.polyval(self.coefficient.flatten(), time)
        # return (self.coefficient.mean() * time * x ** 2).sum()
        
        
class NN(NamedTuple):
    weights: jnp.ndarray
    biases: jnp.ndarray
    
    @classmethod
    def init(
        cls,
        key: jax.random.PRNGKey,
        num_layers: int,
        num_units: int,
    ):
        keys = jax.random.split(key, num_layers)
        weights = [0.01 * jax.random.normal(keys[0], (2, num_units))] + [0.01 * jax.random.normal(k, (num_units, num_units)) for k in keys[1:]]
        biases = [0.01 * jax.random.normal(k, (num_units, )) for k in keys]
        return cls(weights, biases)
        
    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
    ):
        time = time[..., None]
        x = jnp.concatenate([x, time], axis=-1)
        # x = jnp.concatenate([time, time, time], axis=-1)
        for w, b in zip(self.weights, self.biases):
            x = jax.nn.silu(x)
            x = x @ w + b
        x = x.sum(-1)
        # sin = jnp.sin(2 * jnp.pi * time.squeeze(-1))
        # x = x * sin
        return x

        
        
    
    
    