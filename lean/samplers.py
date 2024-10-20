from math import gamma
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Optional
from functools import partial

class HamiltonianMonteCarlo(NamedTuple):
    """Hamiltonian Monte Carlo algorithm.

    Parameters
    ----------
    potential : Callable
        Energy function.

    integrator : Callable
        Integrator function.

    steps: int
        Number of steps to take in the leapfrog integration.
    """
    potential: Callable
    step_size: float
    steps: int

    def step(
            self, 
            position: jnp.ndarray, 
            momentum: jnp.ndarray,
            step: Optional[int] = None,
            reverse: bool = False
        ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
        # compute time
        if step is not None:
            time = (step / self.steps)
        else:
            time = 0.0

        step_size = self.step_size * (-1 if reverse else 1)

        # compose potential energy
        potential = partial(self.potential, time=time)

        # compute force
        force = -jax.grad(potential)(position)

        # leapfrog integration
        # update momentum
        momentum = momentum + 0.5 * step_size * force

        # update position
        position = position + step_size * momentum

        # update force
        force = -jax.grad(potential)(position)

        # update momentum
        momentum = momentum + 0.5 * step_size * force

        return position, momentum
    
    def __call__(
            self,
            position: jnp.ndarray,
            momentum: jnp.ndarray,
    ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
        # initialize state
        state = (position, momentum)

        # run leapfrog integration
        state = jax.lax.fori_loop(
            0, self.steps, lambda i, state: self.step(*state, step=i), state
        )

        # unpack
        position, momentum = state
        return position, momentum
    
    def reverse(
            self,
            position: jnp.ndarray,
            momentum: jnp.ndarray,
    ):
        """Run the Hamiltonian Monte Carlo algorithm in reverse.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.

        schedule : jnp.ndarray
            Schedule for the step size.
        """
        # initialize state
        state = (position, momentum)

        # run leapfrog integration
        state = jax.lax.fori_loop(
            0, self.steps,
            lambda i, state: self.step(*state, step=self.steps-i-1, reverse=True), 
            state,
        )

        # unpack
        position, momentum = state
        return position, momentum



class LangevinDynamics(NamedTuple):
    """Langevin Dynamics algorithm.

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
    gamma: float = 0.9
    beta: float = 1.0
    mass: float = 1.0
    time: float = 1.0
    
    def step(
            self, 
            position: jnp.ndarray, 
            momentum: jnp.ndarray,
            delta_S: float,
            key: jax.random.PRNGKey,
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
        key0, key1 = jax.random.split(key)
        
        # compose potential energy
        potential = lambda x: self.potential(x, time=time).sum()

        # compute force
        force = -jax.grad(potential)(position)
        
        # compute the coefficients
        c1 = self.step_size / (2 * self.mass)
        c2 = jnp.sqrt(4 * self.gamma * self.mass / (self.step_size * self.beta))
        c3 = 1 + self.step_size * self.gamma / 2.0

        # sample noise
        eta = jax.random.normal(key0, shape=momentum.shape)
        
        # leapfrog integration
        # update momentum
        momentum_prime = momentum \
            + c1 * (
                force 
                - self.gamma * self.mass * momentum
                + c2 * eta
            )

        # update position
        position = position + self.step_size * momentum

        # update force
        force = -jax.grad(potential)(position)

        # sample another noise
        eta_prime = jax.random.normal(key1, shape=momentum.shape)

        # update momentum
        momentum_new = (1 / c3) * (
            momentum_prime
            + c1 * (
                force 
                + c2 * eta_prime
            )
        )

        # compute eta tilde
        eta_tilde_prime = eta - \
            jnp.sqrt(self.gamma * self.step_size * self.mass * self.beta) \
                * momentum
        
        eta_tilde = eta_prime - \
            jnp.sqrt(self.gamma * self.step_size * self.mass * self.beta) \
                * momentum_new
                
        # compute delta S
        delta_S = delta_S + 0.5 * (
            (eta ** 2).sum(-1).sum(-1)
            + (eta_prime ** 2).sum(-1).sum(-1)
            - (eta_tilde ** 2).sum(-1).sum(-1)
            - (eta_tilde_prime ** 2).sum(-1).sum(-1)
        ) 
        
        return position, momentum_new, delta_S
    
    def __call__(
            self,
            position: jnp.ndarray,
            momentum: jnp.ndarray,
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
        state = (position, momentum, jnp.zeros(len(position)))

        # run leapfrog integration
        state = jax.lax.fori_loop(
            0, steps, 
            lambda i, state: self.step(*state, time=times[i], key=keys[i]), 
            state,
        )

        # unpack
        position, momentum, delta_S = state
        return position, momentum, delta_S
    
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
            epsilon: float = 1e-2,
            temperature: float = 1.0,
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
        eta_tilde = jnp.sqrt(0.5 * epsilon / temperature) * (
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
    