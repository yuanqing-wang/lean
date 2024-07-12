import sys
import os
from functools import partial
import jax
from jax import numpy as jnp
import numpy as onp
import math
from lean.distributions import RadialLogNormal2D

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
    x = jnp.linalg.norm(x, axis=-1)

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
        schedule_gaussian=None,
        schedule_a=None,
        schedule_b=None,
        schedule_c=None,
        log_sigma=None,
        mu=None,

):
    # gaussian_potential = jax.scipy.stats.norm.logpdf(x, scale=scale).sum(-1).sum(-1).mean()

    gaussian_potential = RadialLogNormal2D(mu, log_sigma).log_prob(x).sum(-1).mean()
    gaussian_potential = (1 - schedule_gaussian(time)) * gaussian_potential

    x = x[..., :, None, :] - x[..., None, :, :]
    x = (x ** 2).sum(-1) + 1e-10
    x = x ** 0.5
    x = x.reshape(*x.shape[:-2], -1)
    a_term = a * (x - d0)
    b_term = b * (x - d0) ** 2
    c_term = c * (x - d0) ** 4

    a_term = a_term * schedule_a(time)
    b_term = b_term * schedule_b(time)
    c_term = c_term * schedule_c(time)
    energy = (1 / (2 * tao)) * (a_term + b_term + c_term).sum(-1).mean()
    energy = energy + gaussian_potential
    return energy

def loss_fn(params, sampler_params, x, key):
    schedules, mu, log_sigma = params
    schedule_gaussian, schedule_a, schedule_b, schedule_c = schedules
    momentum0 = RadialLogNormal2D(mu[0], log_sigma[0]).sample(key, x.shape[:-1])
    _time_dependent_potential = partial(
        time_dependent_potential,
        schedule_gaussian=schedule_gaussian,
        schedule_a=schedule_a,
        schedule_b=schedule_b,
        schedule_c=schedule_c,
        mu=mu[0],
        log_sigma=log_sigma[0],
    )
    sampler = HamiltonianMonteCarlo(
        potential=_time_dependent_potential,
        **sampler_params,
    )
    position, momentum = sampler.reverse(x, momentum0)
    
    loss = -RadialLogNormal2D(mu[0], log_sigma[0]).log_prob(position).sum(-1).mean() \
        - RadialLogNormal2D(mu[1], log_sigma[1]).log_prob(momentum).sum(-1).mean() \
        + RadialLogNormal2D(mu[2], log_sigma[2]).log_prob(momentum0).sum(-1).mean()

    return loss

def run(args):
    data_train = jnp.load(args.data_train)
    data_val = jnp.load(args.data_val)
    data_test = jnp.load(args.data_test)

    key = jax.random.PRNGKey(2666)
    key_gaussian, key_a, key_b, key_c = jax.random.split(key, 4)
    from lean.schedules import SinRBFSchedule
    schedule_gaussian = SinRBFSchedule.init(key_gaussian, 10)
    schedule_a = SinRBFSchedule.init(key_a, 10)
    schedule_b = SinRBFSchedule.init(key_b, 10)
    schedule_c = SinRBFSchedule.init(key_c, 10)
    schedules = [schedule_gaussian, schedule_a, schedule_b, schedule_c]
    scales = jnp.zeros(3)

    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init([schedules, scales])
    sampler_args = {
        'step_size': 1e-3,
        'steps': 100,
    }
    
    key = jax.random.PRNGKey(2666)

    def step(schedules, scales, opt_state, data_train, key):
        subkey, key = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(
            [schedules, scales], sampler_args, data_train, subkey,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        schedules, scales = optax.apply_updates([schedules, scales], updates)
        return schedules, scales, opt_state, loss, key
    
    step = jax.jit(step)
    
    for _ in range(100000):
        schedules, scales, opt_state, loss, key = step(
            schedules, scales, opt_state, data_train, key
        )

        if _ % 100 == 0:
            loss_val = loss_fn([schedules, scales], sampler_args, data_val, key)
            loss_test = loss_fn([schedules, scales], sampler_args, data_test, key)
            print(loss, loss_val, loss_test)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train", type=str, default="data_train.npy")
    parser.add_argument("--data_val", type=str, default="data_val.npy")
    parser.add_argument("--data_test", type=str, default="data_test.npy")
    args = parser.parse_args()
    run(args)