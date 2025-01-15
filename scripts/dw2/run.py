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
N_DIM = 3

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

def gaussian_potential(x):
    return 0.5 * (x ** 2).sum()

def annealing_potential(x, time):
    return (1 - time) * gaussian_potential(x) + time * potential(x)

def ess(log_w):
    # normalize
    w = jax.nn.softmax(log_w)
    ess = 1 / (w ** 2).sum()
    return ess

def loss_fn(unbiasing_potential, position, key):
    integrator = OverdampedLangevinDynamics(
        annealing_potential,
        unbiasing_potential,
        step_size=0.01,
    )
    position, A, B, loss = integrator(position, key)
    jax.debug.print("{position}, {A}", position=position, A=A)
    return loss

def run():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    unbiasing_potential = SinRBF.init(subkey, 10, 10)

    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(unbiasing_potential)
    
    for _ in range(1000):
        key, key0, key1 = jax.random.split(key, 3)
        position = jax.random.normal(key0, (N_SAMPLES, N_PARTICLES, N_DIM))
        loss, grad = jax.value_and_grad(loss_fn)(unbiasing_potential, position, key1)
        jax.debug.print("{x}", x=loss)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        unbiasing_potential = optax.apply_updates(unbiasing_potential, updates)
        
if __name__ == '__main__':
    run()