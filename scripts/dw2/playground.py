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
from lean.loss import loss
from flax.core import FrozenDict


N_SAMPLES = 16
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
    ess = 1 / (w ** 2).sum()
    return ess

def run():
    key = jax.random.PRNGKey(0)
    key0, key1 = jax.random.split(key)
    unbiasing_potential = lambda x, time: 0.0
    integrator = OverdampedLangevinDynamics(
        annealing_potential,
        # potential,
        unbiasing_potential,
        step_size=0.01,
    )
    position = jax.random.normal(key0, (N_SAMPLES, N_PARTICLES, N_DIM))
    position, B = integrator(position, key1)
    print(position.shape)
    print(potential(position[:, -1, :, :]))
    
    B = 0.0
    
    # for idx in range(100):
    #     time = float(idx / 100)
    #     key, subkey = jax.random.split(key)
    #     position, B = integrator.step(position, B, key=subkey, time=time)
    #     # distance = position[..., :, None, :] - position[..., None, :, :]
    #     # distance = (distance ** 2).sum(-1) ** 0.5
    #     # distance = distance[..., 0, 1].flatten()
    #     print(potential(position))
        
    #     # updates, optimizer_state = optimizer.update(grad, optimizer_state)
    #     # unbiasing_potential = optax.apply_updates(unbiasing_potential, updates)
        
if __name__ == '__main__':
    run()