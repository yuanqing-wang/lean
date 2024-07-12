import math
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import NamedTuple

class RadialLogNormal3D(NamedTuple):
    mu: jnp.ndarray
    log_sigma: jnp.ndarray

    def sample(self, key, shape):
        key_r, key_theta, key_phi = jax.random.split(key, 3)
        distribution = tfd.LogNormal(self.mu, self.log_sigma)
        radial = distribution.sample(seed=key_r, shape=shape)
        theta = 2 * jnp.pi * jax.random.uniform(key_theta, shape)
        phi = jnp.arccos(2 * jax.random.uniform(key_phi, shape) - 1)
        x = radial * jnp.sin(phi) * jnp.cos(theta)
        y = radial * jnp.sin(phi) * jnp.sin(theta)
        z = radial * jnp.cos(phi)
        return jnp.stack([x, y, z], axis=-1)

    def log_prob(self, x):
        radial = jnp.linalg.norm(x, axis=-1)
        log_prob = tfd.LogNormal(self.mu, self.log_sigma).log_prob(radial)
        log_prob = log_prob + 2 * jnp.log(1 / (2 * math.pi))
        return log_prob

class RadialLogNormal2D(NamedTuple):
    mu: jnp.ndarray
    log_sigma: jnp.ndarray

    def sample(self, key, shape):
        key_r, key_theta = jax.random.split(key, 3)
        distribution = tfd.LogNormal(self.mu, self.log_sigma)
        radial = distribution.sample(seed=key_r, shape=shape)
        theta = 2 * jnp.pi * jax.random.uniform(key_theta, shape)
        x = radial * jnp.cos(theta)
        y = radial * jnp.sin(theta)
        return jnp.stack([x, y], axis=-1)

    def log_prob(self, x):
        radial = jnp.linalg.norm(x, axis=-1)
        log_prob = tfd.LogNormal(self.mu, self.log_sigma).log_prob(radial)
        log_prob = log_prob + jnp.log(1 / (2 * math.pi))
        return log_prob
