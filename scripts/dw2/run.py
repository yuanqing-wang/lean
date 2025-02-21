import jax
import jax.numpy as jnp
import sys
import os
from functools import partial
import jax
import optax
from jax import numpy as jnp
import numpy as onp
import math
from lean.samplers import OverdampedLangevinDynamics
from lean.unbiasing import SinRBF
from flax.core import FrozenDict


N_SAMPLES = 1000
N_PARTICLES = 4
N_DIM = 2

def potential(
        x, 
        tao=1.0, 
        a=0.0, 
        b=-4.0, 
        c=0.9, 
        d0=4.0,
        **kwargs,
):

    x = x[..., :, None, :] - x[..., None, :, :]
    x = (x ** 2).sum(-1) 
    is_zero = x == 0.0
    x = x + 1e-5
    x = x ** 0.5

    energy = (1 / (2 * tao)) * (
        a * (x - d0)
        + b * (x - d0) ** 2
        + c * (x - d0) ** 4
    )
    
    energy = jnp.where(is_zero, 0.0, energy).sum()
    return energy

# from typing import NamedTuple
# class Schedule(NamedTuple):
#     k: jnp.ndarray
#     b: jnp.ndarray
    
#     @classmethod
#     def init(
#         cls,
#         key: jax.random.PRNGKey,
#     ):
#         k = jax.random.normal(key)
#         b = jax.random.normal(key)
#         return cls(k, b)
    
#     def __call__(self, x, time):
#         position = time * self.k + self.b
#         sin = jnp.sin(time * jnp.pi)
#         return sin * 0.5 * ((x - position) ** 2).sum()

# def potential(x):
#     return 0.5 * ((x - 1) ** 2).sum()

def gaussian_potential(x):
    return 0.5 * (x ** 2).sum()

def annealing_potential(x, time):
    return (1 - time) * gaussian_potential(x) + time * potential(x)

def ess(log_w):
    # normalize
    w = jax.nn.softmax(log_w)
    ess = 1 / (w ** 2).sum()
    return ess

@jax.jit
def loss_fn(unbiasing_potential, position, key):
    integrator = OverdampedLangevinDynamics(
        annealing_potential,
        unbiasing_potential,
        step_size=0.01,
        time=1.0,
    )
    position, A, B, loss = integrator(position, key)
    return loss, (A, position)

def run():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    unbiasing_potential = SinRBF.init(subkey, 10, 10)
    # unbiasing_potential = Schedule.init(subkey)
    # unbiasing_potential = lambda x, t: 0.0

    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(unbiasing_potential)
    
    for _ in range(100000):
        key, key0, key1 = jax.random.split(key, 3)
        position = jax.random.normal(key0, (N_SAMPLES, N_PARTICLES, N_DIM))
        
        (loss, (A, position)), grad = jax.value_and_grad(loss_fn, has_aux=True)(unbiasing_potential, position, key1)
        ESS = ess(A)
        print(ESS, loss)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        unbiasing_potential = optax.apply_updates(unbiasing_potential, updates)
        
if __name__ == '__main__':
    run()