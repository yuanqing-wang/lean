import jax
import jax.numpy as jnp
import sys
import os
from functools import partial
import jax
from jax import numpy as jnp
import numpy as onp
import math
from lean.distributions import CenteredNormal
from flax.core import FrozenDict
from scripts.dw2.run_ld import N_PARTICLES


N_SAMPLES = 100
N_PARTICLES = 4
N_DIM = 2

def potential(
        x, 
        epsilon=1.0,
        tau=1.0,
        r=1.0,
        **kwargs,
):
    x = x[..., :, None, :] - x[..., None, :, :]
    x = jax.nn.relu((x ** 2).sum(-1))
    x = x ** 0.5
    x = jnp.where(x > 0, r/x, 0.0)

    energy = (0.5 * epsilon / tau) \
        * (x ** 12 - 2 * x ** 6)
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
        epsilon=1.0,
        tau=1.0,
        r=1.0,
        time=0.0,
        schedule_gaussian=None,
        schedule6=None,
        schedule12=None,
        log_sigma=None,

):
    # gaussian_potential = jax.scipy.stats.norm.logpdf(x, scale=scale).sum(-1).sum(-1).mean()

    gaussian_potential = CenteredNormal(log_sigma).log_prob(x).sum(-1).sum(-1).mean()
    gaussian_potential = (1 - schedule_gaussian(time)) * gaussian_potential

    x = x[..., :, None, :] - x[..., None, :, :]
    x = jax.nn.relu((x ** 2).sum(-1))
    x = x ** 0.5
    x = jnp.where(x > 0, r/x, 0.0)

    term6 = -epsilon / tau * x ** 3
    term12 = 0.5 * epsilon / tau * x ** 2
    
    term6 = term6 * schedule6(time)
    term12 = term12 * schedule12(time)
    energy = term6 + term12
    energy = energy + gaussian_potential
    return energy


from typing import Callable, NamedTuple, Optional
class OverdampedLangevinDynamics(NamedTuple):
    """Overdamped Langevin Dynamics.

    Parameters
    ----------
    potential : Callable
        Energy function.

    integrator : Callable
        Integrator function.

    steps: int
        Number of steps to ta
        
    """
        
    potential: Callable
    step_size: float
    time: float = 1.0
    
    def step(
            self, 
            position: jnp.ndarray, 
            delta_S: float,
            key: jax.random.PRNGKey,
            epsilon: float = 1.0,
            temperature: float = 0.0,
            time: float = 0.0,
    ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
        
        # compose potential energy
        potential = lambda x: self.potential(x, time=time).sum()

        # compute force
        force = -jax.grad(potential)(position)
        
        # sample noise
        eta = jax.random.normal(key, shape=position.shape)
        
        position = position \
            + epsilon * force + jnp.sqrt(2 * epsilon * temperature) * eta
        
        new_force = -jax.grad(potential)(position)
        eta_tilde = jnp.sqrt(0.5 * temperature * epsilon) * (
            force + new_force
        ) - eta
                
        # compute delta S
        delta_S = delta_S + 0.5 * (
            (eta ** 2).sum(-1).sum(-1)
            - (eta_tilde ** 2).sum(-1).sum(-1)
        ) 
        
        return position, delta_S
    
    def __call__(
            self,
            position: jnp.ndarray,
            key: jax.random.PRNGKey,
    ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """        
        # split keys
        steps = int(self.time / self.step_size)
        keys = jax.random.split(key, steps)
        times = jnp.linspace(0, 1, steps)
        
        # initialize state
        state = (position, jnp.zeros(len(position)))

        # run leapfrog integration
        state = jax.lax.fori_loop(
            0, steps, 
            lambda i, state: self.step(*state, time=times[i], key=keys[i]), 
            state,
        )

        # unpack
        position, delta_S = state
        return position, delta_S
    
def compute_log_w(schedules, sampler_params, key):
    schedule_gaussian, schedule6, schedule12 = schedules
    key, key_x = jax.random.split(key, 2)
    x = CenteredNormal(0.0).sample(key_x, (N_SAMPLES, N_PARTICLES, N_DIM))
    h = jnp.zeros((N_SAMPLES, N_PARTICLES, N_DIM))
    
    _time_dependent_potential = partial(
        time_dependent_potential,
        schedule_gaussian=schedule_gaussian,
        schedule6=schedule6,
        schedule12=schedule12,
        log_sigma=0.0,
    )
    
    sampler = OverdampedLangevinDynamics(
        potential=_time_dependent_potential,
        **sampler_params,
    )
    
    x, delta_S = sampler(x, key=key)
    log_w = -potential(x).sum(-1).sum(-1) + CenteredNormal(0.0).log_prob(x).sum(-1).sum(-1) + delta_S
    return log_w

def loss_fn(schedules, sampler_params, key):
    log_w = compute_log_w(schedules, sampler_params, key)
    w = jax.nn.logsumexp(log_w)
    return -w.mean()

def run():
    key = jax.random.PRNGKey(1984)
    key_gaussian, key_a, key_b, key_c = jax.random.split(key, 4)
    from lean.schedules import SinRBFSchedule
    schedule_gaussian = SinRBFSchedule.init(key_gaussian, 100)
    schedule_6 = SinRBFSchedule.init(key_a, 100)
    schedule_12 = SinRBFSchedule.init(key_b, 100)
    schedules = [schedule_gaussian, schedule_6, schedule_12]

    import optax
    optimizer = optax.adam(1e-5)
    opt_state = optimizer.init(schedules)
    sampler_args = {
        'step_size': 0.01,
        'time': 1.0,
    }
    sampler_args = FrozenDict(sampler_args)

    @jax.jit
    def step(schedules, opt_state, key):
        key, subkey = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(
            schedules, sampler_args, subkey,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        schedules = optax.apply_updates(schedules, updates)
        return schedules, opt_state, key

    for idx in range(10000000):
        schedules, opt_state, key = step(schedules, opt_state, key)
        
        if idx % 1000 == 0:
            # eval
            log_w = compute_log_w(schedules, sampler_args, key)
            print(f'ESS: {ess(log_w)}')
        
if __name__ == '__main__':
    run()