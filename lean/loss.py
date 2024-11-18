import jax
from jax import numpy as jnp

def loss(unbiasing_potential, trajectory, A, time):
    trajectory = jax.lax.stop_gradient(trajectory)
    A = jax.nn.softmax(A, 0)
    dt_potential = jax.vmap(
        jax.vmap(
            lambda x, t: jax.grad(unbiasing_potential, argnums=1)(x, t), 
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )(trajectory, time)
            
    force_norm_sq = jax.vmap(
        lambda x, t: (jax.grad(unbiasing_potential, argnums=0)(x, t) ** 2).sum(-1).sum(-1),
        in_axes=(1, 0),
    )(trajectory, time).transpose()
        
    return jnp.mean(A * (force_norm_sq * 0.5 + dt_potential))
    

    
    