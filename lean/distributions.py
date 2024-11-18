import math
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import NamedTuple

class CenteredNormal(NamedTuple):
    log_sigma: jnp.ndarray
    
    def sample(self, key, shape):
        x = jax.random.normal(key, shape) * jnp.exp(self.log_sigma)
        x = x - x.mean(axis=-2, keepdims=True)
        return x
    
    def log_prob(self, x):
        x = x - x.mean(axis=-2, keepdims=True)
        N = x.shape[-2]
        D = x.shape[-1]
        DoF = D * (N - 1)
        normalizing_constant = -0.5 * DoF * jnp.log(2*math.pi) - 0.5 * self.log_sigma
        log_prob = normalizing_constant - 0.5 * (x / jnp.exp(self.log_sigma)) ** 2
        return log_prob
        