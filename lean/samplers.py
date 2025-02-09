from math import gamma
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Optional
from functools import partial
    
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
    unbiasing_potential: Callable
    step_size: float
    time: float = 1.0
    
    def step(
            self, 
            position: jnp.ndarray, 
            A: float,
            B: float,
            loss: float,
            key: jax.random.PRNGKey,
            epsilon: float = 5.0,
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
        position = jax.lax.stop_gradient(position)
        time = time * jnp.ones(len(position))
        
        # compose potential energy
        dx_f, dt_f = jax.vmap(jax.grad(self.unbiasing_potential, argnums=(0, 1)))(position, time)
        dx_u, dt_u = jax.vmap(jax.grad(self.potential, argnums=(0, 1)))(position, time)
        
        # sample noise
        eta = jax.random.normal(key, shape=position.shape)
        
        # update position
        position = position \
            - epsilon * dx_u * self.step_size \
            + dx_f * self.step_size \
            + jnp.sqrt(2 * epsilon) * eta * self.step_size
                        
                        
        # update B                    
        B = B \
            + (1 / epsilon) * (dx_f ** 2).sum(-1).sum(-1) * self.step_size \
            + jnp.sqrt(2 / epsilon) * (self.step_size * (dx_f * eta)).sum(-1).sum(-1) \
            + dt_u * self.step_size \
            + (1 / epsilon) * dt_f * self.step_size
                    
        # A = -jax.vmap(self.potential)(position, time) - B
        A = (1 / epsilon) * jax.vmap(self.unbiasing_potential)(position, time) - B
        
        _loss = jax.nn.softmax(A, 0) * (0.5 * (dx_f ** 2).sum(-1).sum(-1) + dt_f)
        _loss = _loss.mean()
        loss = loss + _loss
        
        return position, A, B, loss
            
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
        state = (position, jnp.zeros(len(position)), jnp.zeros(len(position)), 0.0)
        
        # def step_fn(state, idx):
        #     state = self.step(*state, time=times[idx], key=keys[idx])
        #     return state, state
        # 
        # _, states = jax.lax.scan(
        #     step_fn,
        #     state,
        #     jnp.arange(steps),
        # )
        # 
        # # unpack
        # position, A, _ = states
        # position = position.swapaxes(0, 1)
        # A = A.swapaxes(0, 1)
        # return position, A
        
        def step_fn(idx, state):
            state = self.step(*state, time=times[idx], key=keys[idx])
            return state
        
        state = jax.lax.fori_loop(0, steps, step_fn, state)
        return state


    