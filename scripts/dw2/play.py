import jax
import jax.numpy as jnp
import sys
import os
from functools import partial
import jax
from lean import unbiasing
import optax
from jax import numpy as jnp
import numpy as onp
import math
from lean.samplers import OverdampedLangevinDynamics
from lean.unbiasing import SinRBF
from lean.loss import loss
from flax.core import FrozenDict


N_SAMPLES = 100
N_PARTICLES = 2
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

def gaussian_potential(x):
    return 0.5 * (x ** 2).sum()

def annealing_potential(x, time):
    return (1 - time) * gaussian_potential(x) + time * potential(x)

def ess(log_w):
    # normalize
    w = jax.nn.softmax(log_w)
    ess = 1 / ((w ** 2).sum())
    return ess


def run():
    global loss
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    # unbiasing_potential = SinRBF.init(subkey, 10, 10)
    unbiasing_potential = lambda x, t: 0.0

    optimizer = optax.adam(1e-5)
    # optimizer_state = optimizer.init(unbiasing_potential)
    
    for _ in range(1):
        
        key, key0, key1 = jax.random.split(key, 3)
        integrator = OverdampedLangevinDynamics(
            annealing_potential,
            unbiasing_potential,
            step_size=0.001,
            time=1.0
        )
        position = jax.random.normal(key0, (N_SAMPLES, N_PARTICLES, N_DIM))
        position, A, B, loss = integrator(position, key1)
        print(ess(A))
        
        # updates, optimizer_state = optimizer.update(grad, optimizer_state)
        # unbiasing_potential = optax.apply_updates(unbiasing_potential, updates)
        
if __name__ == '__main__':
    run()