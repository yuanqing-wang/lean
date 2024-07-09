import sys
import os
from functools import partial
import jax
from jax import numpy as jnp
import numpy as onp
import math

sys.path.append(os.path.abspath("en_flows"))
from lean.samplers import HamiltonianMonteCarlo

def potential(
        x, 
        tao=1.0, 
        a=0.0, 
        b=-4.0, 
        c=0.9, 
        d0=4.0,
        **kwargs,
):
    energy = (1 / (2 * tao)) * (
        a * (x - d0)
        + b * (x - d0) ** 2
        + c * (x - d0) ** 4
    )
    return energy.sum()

def cos(t):
    return jnp.cos(0.5 * t / math.pi)

def sin(t):
    return jnp.sin(0.5 * t / math.pi)

def coefficient(t, a, b):
    return jax.scipy.special.betainc(a, b, t)

def time_dependent_potential(
        x, 
        tao=1.0, 
        a=0.0, 
        b=-4.0, 
        c=0.9, 
        d0=4.0,
        time=0.0,
        schedule=None,
):
    gaussian_potential = jax.scipy.stats.norm.logpdf(x).sum(-1).sum(-1).mean()
    gaussian_potential = coefficient(1 - time, schedule[0], schedule[1]) * gaussian_potential

    a_term = a * (x - d0)
    b_term = b * (x - d0) ** 2
    c_term = c * (x - d0) ** 4

    a_term = a_term * coefficient(time, schedule[2], schedule[3])
    b_term = b_term * coefficient(time, schedule[4], schedule[5])
    c_term = c_term * coefficient(time, schedule[6], schedule[7])

    energy = (1 / (2 * tao)) * (a_term + b_term + c_term)
    return energy.sum()

def loss_fn(schedule, sampler_params, x, key):
    momentum0 = jax.random.normal(key, x.shape)
    _time_dependent_potential = partial(
        time_dependent_potential,
        schedule=schedule,
    )
    sampler = HamiltonianMonteCarlo(
        potential=_time_dependent_potential,
        **sampler_params,
    )
    position, momentum = sampler.reverse(x, momentum0)
    loss = -jax.scipy.stats.norm.logpdf(position).sum(-1).sum(-1).mean() \
        - jax.scipy.stats.norm.logpdf(momentum).sum(-1).sum(-1).mean() \
        + jax.scipy.stats.norm.logpdf(momentum0).sum(-1).sum(-1).mean()
    return loss

def run(args):
    data_train = jnp.load(args.data_train)
    data_val = jnp.load(args.data_val)
    data_test = jnp.load(args.data_test)

    schedule = jnp.ones(8)

    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(schedule)
    sampler_args = {
        'step_size': 1e-3,
        'steps': 100,
    }
    
    key = jax.random.PRNGKey(2666)

    def step(schedule, opt_state, data_train, key):
        subkey, key = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(
            schedule, sampler_args, data_train, subkey,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        schedule = optax.apply_updates(schedule, updates)
        return schedule, opt_state, loss, key
    
    step = jax.jit(step)
    
    for _ in range(100000):
        schedule, opt_state, loss, key = step(
            schedule, opt_state, data_train, key
        )
        print(loss)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train", type=str, default="data_train.npy")
    parser.add_argument("--data_val", type=str, default="data_val.npy")
    parser.add_argument("--data_test", type=str, default="data_test.npy")
    args = parser.parse_args()
    run(args)