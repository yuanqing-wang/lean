import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Optional
from functools import partial
import chex
    
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
    steps: int
    time: float = 1.0
    
    def step(
            self, 
            position: jnp.ndarray, 
            A: float,
            B: float,
            loss: float,
            time: float,
            step_size: float,
            f0: float,
            key: jax.random.PRNGKey,
            epsilon: float = 1.0,
    ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
        time = time * jnp.ones((len(position), ))
        
        # compose potential energy
        dx_f, dt_f = jax.grad(lambda x, t: self.unbiasing_potential(x, t).sum(), argnums=(0, 1))(position, time)
        dx_u, dt_u = jax.grad(lambda x, t: self.potential(x, t).sum(), argnums=(0, 1))(position, time)
        chex.assert_shape(dx_f, position.shape)
        chex.assert_shape(dx_u, position.shape)
        chex.assert_shape(dt_f, (len(position), ))
        chex.assert_shape(dt_u, (len(position), ))
                
        # sample noise
        eta = jax.random.normal(key, shape=position.shape)
        
        # update position
        position = position \
            - epsilon * dx_u * step_size \
            + dx_f * step_size \
            + jnp.sqrt(2 * epsilon * step_size) * eta
                        
        # update B                            
        B = B \
            + dt_u * step_size \
            + (1 / epsilon) * dt_f * step_size \
            + (1 / epsilon) * (dx_f ** 2).sum(-1) * step_size \
            + jnp.sqrt(2 * step_size / epsilon) * (dx_f * eta).sum(-1)
        
        A = (1 / epsilon) * (
            self.unbiasing_potential(position, time)
            - f0
        ) 
        
        chex.assert_shape(A, (len(position), ))
        chex.assert_shape(A, (len(position), ))
        
        A = A - B

        _loss = jax.nn.softmax(A, 0) * (0.5 * (dx_f ** 2).sum(-1) + dt_f)
        
        chex.assert_equal_shape([A, _loss])
        
        _loss = _loss.sum() * step_size
                
        position = jax.lax.stop_gradient(position)
        A = jax.lax.stop_gradient(A)
        B = jax.lax.stop_gradient(B)
                                
        loss = loss + _loss
        return position, A, B, loss
            
    def __call__(
            self,
            position: jnp.ndarray,
            key: jax.random.PRNGKey,
            f0: jnp.ndarray,
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
        steps = self.steps
        keys = jax.random.split(key+1, steps)
        times = jax.random.uniform(keys[-1], shape=(steps,)) * self.time
        times = jnp.sort(times)
        times = jnp.concatenate([times, jnp.array([self.time])])
        # step_sizes = times[1:] - times[:-1]
        step_size = self.time / steps
        
        # initialize state
        state = (position, jnp.zeros(len(position)), jnp.zeros(len(position)), 0.0)
                
        def step_fn(idx, state):
            state = self.step(*state, time=times[idx], key=keys[idx], step_size=step_size, f0=f0)
            return state
        
        state = jax.lax.fori_loop(0, steps, step_fn, state)                
        
        # for idx in range(steps):
        #     state = step_fn(idx, state)
        
        position, A, B, loss = state
        return position, A, B, loss
        
        


    