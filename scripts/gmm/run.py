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

N_SAMPLES = 100

try:
    from fabjax.targets.gmm_v0 import GMM
    gmm = GMM()
except:
    from fabjax.targets.gmm_v0 import GMM
    gmm = GMM()

def potential(x):
    return -gmm.log_prob(x)

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
def loss_fn(unbiasing_potential, position, key, time):
    f0 = unbiasing_potential(position, jnp.zeros(len(position)))
    integrator = OverdampedLangevinDynamics(
        annealing_potential,
        unbiasing_potential,
        steps=100,
        time=time,
    )
    
    position, A, B, loss = integrator(position, key, f0=f0)
    f1 = unbiasing_potential(position, time * jnp.ones(len(position)))
    f1 = (jax.nn.softmax(A, 0) * f1).sum()
    f0 = f0.mean()
    loss = loss + f0 - f1
    return loss, (A, position)

def run():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    unbiasing_potential = NN.init(subkey, 3, 20)
    optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    
    optimizer_state = optimizer.init(unbiasing_potential)
    
    for idx in range(100000):
        key, key0, key1 = jax.random.split(key, 3)
        position = jax.random.normal(key0, (N_SAMPLES, 2))
        (loss, (A, position)), grad = jax.value_and_grad(loss_fn, has_aux=True)(unbiasing_potential, position, key1, time=1.0)
        ESS = ess(A)
        print(ESS, loss)
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params=unbiasing_potential)
        unbiasing_potential = optax.apply_updates(unbiasing_potential, updates)
        
        
if __name__ == '__main__':
    run()