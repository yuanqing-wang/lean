import sys
import os
from functools import partial
import jax
from jax import numpy as jnp
import numpy as onp
import math
from lean.distributions import CenteredNormal
from flax.core import FrozenDict

sys.path.append(os.path.abspath("en_flows"))
from lean.samplers import HamiltonianMonteCarlo, LangevinDynamics

N_PARTICLES = 10

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
    x = (x ** 2).sum(-1) + 1e-10
    x = x ** 0.5

    energy = (1 / (2 * tao)) * (
        a * (x - d0)
        + b * (x - d0) ** 2
        + c * (x - d0) ** 4
    )
    return energy

def cos(t):
    return jnp.cos(0.5 * t / math.pi)

def sin(t):
    return jnp.sin(0.5 * t / math.pi)

def coefficient(t, a, b):
    return jax.scipy.special.betainc(a, b, t)

def ess(log_w):
    # normalize
    w = jax.nn.softmax(log_w)
    ess = 1 / (w ** 2).sum()
    return ess

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

):
    # gaussian_potential = jax.scipy.stats.norm.logpdf(x, scale=scale).sum(-1).sum(-1).mean()

    gaussian_potential = CenteredNormal(log_sigma).log_prob(x).sum(-1).sum(-1).mean()
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

@partial(jax.jit, static_argnums=(1, 3))
def compute_log_w(schedules, sampler_params, key, model):
    schedule_gaussian, schedule_a, schedule_b, schedule_c, nn_params = schedules
    key, key_x, key_v = jax.random.split(key, 3)
    x = CenteredNormal(0.0).sample(key_x, (N_PARTICLES, 2, 1))
    h = jnp.zeros((N_PARTICLES, 2, 1))
    h, v = model.apply(nn_params, h, x)
    
    _time_dependent_potential = partial(
        time_dependent_potential,
        schedule_gaussian=schedule_gaussian,
        schedule_a=schedule_a,
        schedule_b=schedule_b,
        schedule_c=schedule_c,
        log_sigma=0.0,
    )
    
    sampler = LangevinDynamics(
        potential=_time_dependent_potential,
        **sampler_params,
    )
    
    x, v, delta_S = sampler(x, v, key=key)
    print(potential(x).shape)
    log_w = -potential(x).sum(-1).sum(-1) # + CenteredNormal(0.0).log_prob(x).sum(-1).sum(-1) + delta_S
    jax.debug.print("{x}", x=log_w)
    return log_w

def loss_fn(schedules, sampler_params, key, model):
    log_w = compute_log_w(schedules, sampler_params, key, model)
    return -log_w.mean()
    
def run(args):
    from lean.models import EGNNModel
    model = EGNNModel(16, 3)
    nn_params = model.init(jax.random.PRNGKey(2666), jnp.zeros((1, 4, 1)), jnp.zeros((1, 4, 2)))
    key = jax.random.PRNGKey(1984)
    key_gaussian, key_a, key_b, key_c = jax.random.split(key, 4)
    from lean.schedules import SinRBFSchedule
    schedule_gaussian = SinRBFSchedule.init(key_gaussian, 100)
    schedule_a = SinRBFSchedule.init(key_a, 100)
    schedule_b = SinRBFSchedule.init(key_b, 100)
    schedule_c = SinRBFSchedule.init(key_c, 100)
    schedules = [schedule_gaussian, schedule_a, schedule_b, schedule_c, nn_params]
    
    import optax
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(schedules)
    sampler_args = {
        'step_size': 0.01,
        'beta': 1e-1,
        # 'gamma': 1e-3,
    }
    sampler_args = FrozenDict(sampler_args)
    
    @partial(jax.jit, static_argnums=(3,))
    def step(schedules, opt_state, key, model):
        key, subkey = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(
            schedules, sampler_args, subkey,model=model,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        schedules = optax.apply_updates(schedules, updates)
        return schedules, opt_state, key
    
    for idx in range(10000000):
        schedules, opt_state, key = step(schedules, opt_state, key, model=model)
        
        # if idx % 1000 == 0:
        #     # eval
        #     log_w = compute_log_w(schedules, sampler_args, key, model=model)

        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train", type=str, default="data_train.npy")
    parser.add_argument("--data_val", type=str, default="data_val.npy")
    parser.add_argument("--data_test", type=str, default="data_test.npy")
    args = parser.parse_args()
    run(args)