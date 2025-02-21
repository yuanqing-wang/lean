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

def u0(x):
    return (x**2).sum()

def u1(x):
    return ((x-1)**2).sum()

def annealing_potential(x, t):
    return (1 - t) * u0(x) + t * u1(x)

def get_trajectory():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (100, 1, 1))
    integrator = OverdampedLangevinDynamics(
        annealing_potential,
        lambda x, t: 0.0,
        step_size=0.001,
        time=1.0
    )
    xs = []
    for time in range(1000):
        time = float(time) / 1000
        key, subkey = jax.random.split(key)
        x, _, __, ___ = integrator.step(
            position=x,
            A=0.0,
            B=0.0,
            loss=0.0,
            key=key,
            epsilon=100.0,
            time=time
        )
        xs.append(x)
    xs = jnp.stack(xs, axis=1)
    return xs

def run():
    xs = get_trajectory()
    print(xs.shape)
    import matplotlib.pyplot as plt
    for idx in range(10):
        plt.plot(xs[idx].flatten())
    plt.savefig("toy.png")
    
if __name__ == "__main__":
    run()

