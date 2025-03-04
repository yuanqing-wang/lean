import jax
import jax.numpy as jnp
import sys
import os
from functools import partial
import jax
from lean import unbiasing
import optax
from jax import numpy as jnp
import numpy as onp
import math
from lean.samplers import OverdampedLangevinDynamics

def u0(x):
    return (x**2).sum()

def u1(x):
    return ((x-1)**2).sum()

def annealing_potential(x, t):
    return (1 - t) * u0(x) + t * u1(x)

def get_trajectory():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (100, 1))
    integrator = OverdampedLangevinDynamics(
        annealing_potential,
        lambda x, t: 0.0,
        step_size=0.001,
        time=1.0
    )
    xs = []
    for time in range(1000):
        time = float(time) / 1000
        key, subkey = jax.random.split(key)
        x, _, __, ___, ____, _____= integrator.step(
            position=x,
            A=0.0,
            B=0.0,
            loss=0.0,
            key=key,
            epsilon=100.0,
            time=time
        )
        xs.append(x)
    xs = jnp.stack(xs, axis=1)
    return xs

from flax import linen as nn
class MLP(nn.Module):
  num_hid : int
  num_out : int

  def setup(self):
    self.linear1 = nn.Dense(features=self.num_hid)
    self.linear2 = nn.Dense(features=self.num_hid)
    self.linear3 = nn.Dense(features=self.num_hid)
    self.linear4 = nn.Dense(features=self.num_out)

  def __call__(self, x, t):
    h = jnp.hstack([t,x])
    h = self.linear1(h)
    h = nn.relu(h)
    h = self.linear2(h)
    h = nn.swish(h)
    h = self.linear3(h)
    h = nn.swish(h)
    h = self.linear4(h)
    return h.sum()

from jax import random
import numpy as np
def diamonds(key, t):
    bs = t.shape[0]
    keys = random.split(key, 3)
    x_1 = random.randint(keys[0], minval=0, maxval=2, shape=(bs, 2))
    x_1 = x_1.astype(np.float32)-0.5
    x_1 += 5e-1*(random.uniform(keys[1], shape=(bs,2))-0.5)
    x_0 = 5e-1*(random.uniform(keys[2], shape=(bs,2))-0.5)
    x_t = (1-t)*x_0 + t*x_1
    R = jnp.array([[1/math.sqrt(2),-1/math.sqrt(2)],
                    [1/math.sqrt(2),1/math.sqrt(2)]])
    return x_t@R

def run():
    # xs = get_trajectory()
    # print(xs.shape)
    # import matplotlib.pyplot as plt
    # for idx in range(10):
    #     plt.plot(xs[idx].flatten())
    # plt.savefig("toy.png")
    t = jnp.linspace(0, 1, 1000)[None, :, None]
    t = jnp.repeat(t, 100, axis=0)
    xs = jax.vmap(diamonds)(jax.random.split(jax.random.PRNGKey(0), 100), t)
    
    unbiasing_potential = MLP(128, 1)
    params = unbiasing_potential.init(jax.random.PRNGKey(0), 0.0, xs[0, 0, :])
    
    optimizer = optax.adam(1e-2)
    optimizer_state = optimizer.init(params)
    

    # @jax.jit
    def loss(params, key):
        # t = jax.random.uniform(key, (100, 1000, 1), minval=0.0, maxval=1.0)
        t = jax.random.uniform(key, (100, 1), minval=0.0, maxval=1.0)
        xs = diamonds(key, t)
        dynamics = lambda x, t: unbiasing_potential.apply(params, x, t)
        # dx_f, dt_f = jax.vmap(jax.vmap(jax.grad(dynamics, argnums=(0, 1))))(xs, t)
        dx_f, dt_f = jax.grad(lambda x, t: dynamics(x, t).sum(), argnums=(0, 1))(xs, t)
        
        x0 = diamonds(key, jnp.zeros([100, 1]))
        x1 = diamonds(key, jnp.ones([100, 1]))
        f0 = unbiasing_potential.apply(params, x0, jnp.zeros([100, 1]))
        f1 = unbiasing_potential.apply(params, x1, jnp.ones([100, 1]))
        loss = (f0 - f1) + (dx_f**2).sum() + dt_f.sum()
        return loss
    
    def sample_t(u0, n, t0=0.0, t1=1.0):
        u = (u0 + math.sqrt(2)*np.arange(n + 1)) % 1
        u = u.reshape([-1,1])
        return u[:-1]*(t1-t0) + t0, u[-1]
    
    def am_loss(params, key):        
        keys = random.split(key, 3)
        # boundaries
        t_0, t_1 = jnp.zeros([100, 1]), jnp.ones([100, 1])
        x_0, x_1 = diamonds(keys[0], t_0), diamonds(keys[1], t_1)
        loss = unbiasing_potential.apply(params, x_0, t_0) - unbiasing_potential.apply(params, x_1, t_1)
        dsdtdx = jax.grad(lambda p, t, x: unbiasing_potential.apply(p,x,t).sum(), argnums=[1,2])
        
        # time
        # t, u0 = sample_t(u0, 100)
        t = random.uniform(keys[2], shape=(100, 1), minval=0.0, maxval=1.0)
        # t = jnp.linspace(0, 1, 100)[:, None]
        x_t = diamonds(keys[2], t)
        dsdt, dsdx = dsdtdx(params, t, x_t)            
        loss = loss + dsdt.sum() + 0.5*(dsdx**2).sum(1, keepdims=True).sum()
        return loss.mean()
    
    for i in range(1000000):
        key = jax.random.PRNGKey(i)
        # (_loss, u0), grad = jax.value_and_grad(am_loss, has_aux=True)(params, key, u0)
        (_loss), grad = jax.value_and_grad(loss)(params, key)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        params = optax.apply_updates(params, updates)
        print(_loss)
    
    
if __name__ == "__main__":
    run()

