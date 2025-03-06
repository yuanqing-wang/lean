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
from lean.unbiasing import SinRBF, NN 
from flax.core import FrozenDict


N_SAMPLES = 1000

def potential(
        x, 
        tao=1.0, 
        a=0.0, 
        b=-4.0, 
        c=0.9, 
        d0=4.0,
        **kwargs,
):
    
    x = jnp.abs(x)
    energy = (1 / (2 * tao)) * (
        a * (x - d0)
        + b * (x - d0) ** 2
        + c * (x - d0) ** 4
    )
    
    energy = energy.sum(-1)
    return energy

def gaussian_potential(x):
    return 0.5 * (x ** 2).sum(-1)

def annealing_potential(x, time):
    return (1 - time) * gaussian_potential(x) + time * potential(x)

def ess(log_w):
    # normalize
    w = jax.nn.softmax(log_w)
    ess = 1 / (w ** 2).sum()
    return ess

@jax.jit
def loss_fn(unbiasing_potential, position, key):
    unbiasing_potential = partial(unbiasing_potential)
    integrator = OverdampedLangevinDynamics(
        annealing_potential,
        unbiasing_potential,
        steps=100,
        time=1.0,
    )
    
    position, A, B, loss = integrator(position, key)
    return loss, (A, position)

def run():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    # unbiasing_potential = SinRBF.init(subkey, 20, 20)
    unbiasing_potential = NN.init(subkey, 3, 20)
    optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    
    optimizer_state = optimizer.init(unbiasing_potential)
    
    for idx in range(100):
        # T = float(idx+1) / 100000
        key, key0, key1 = jax.random.split(key, 3)
        position = jax.random.normal(key0, (N_SAMPLES, 1))
        (loss, (A, position)), grad = jax.value_and_grad(loss_fn, has_aux=True)(unbiasing_potential, position, key1)
        ESS = ess(A)
        print(ESS, loss)
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params=unbiasing_potential)
        unbiasing_potential = optax.apply_updates(unbiasing_potential, updates)
        
        
    import pickle
    with open('unbiasing_potential.pkl', 'wb') as f:
        pickle.dump(unbiasing_potential, f)
        
if __name__ == '__main__':
    run()