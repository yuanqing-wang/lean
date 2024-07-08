import sys
import os

import jax
from jax import numpy as jnp
import numpy as onp

sys.path.append(os.path.abspath("en_flows"))

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

def time_dependent_potential(
        x, 
        tao=1.0, 
        a=0.0, 
        b=-4.0, 
        c=0.9, 
        d0=4.0,
        time=0.0,
        a_coefficient_sin=0.5,
        b_coefficient_sin=0.5,
        c_coefficient_sin=0.5,
        a_coefficient_cos=0.5,
        b_coefficient_cos=0.5,
        c_coefficient_cos=0.5,
):
    energy = (1 / (2 * tao)) * (
        a * (x - d0)
        + b * (x - d0) ** 2
        + c * (x - d0) ** 4
    )

    a_term = a * (x - d0)
    b_term = b * (x - d0) ** 2
    c_term = c * (x - d0) ** 4

    a_term = a_term * (a_coefficient_sin * jnp.sin(time) + a_coefficient_cos * jnp.cos(time))
    b_term = b_term * (b_coefficient_sin * jnp.sin(time) + b_coefficient_cos * jnp.cos(time))
    c_term = c_term * (c_coefficient_sin * jnp.sin(time) + c_coefficient_cos * jnp.cos(time))

    energy = (1 / (2 * tao)) * (a_term + b_term + c_term)
    return energy.sum()

def run(args):
    from en_flows.dw4_experiment.dataset import get_data_dw4, remove_mean
    data_train, batch_iter_train = get_data_dw4(args.n_data, 'train', 100)
    data_val, batch_iter_val = get_data_dw4(args.n_data, 'val', 100)
    data_test, batch_iter_test = get_data_dw4(args.n_data, 'test', 100)

    data_train = data_train.reshape(-1, 4, 2)
    data_val = data_val.reshape(-1, 4, 2)
    data_test = data_test.reshape(-1, 4, 2)

    data_train = data_train - data_train.mean(dim=-2, keepdim=True)
    data_val = data_val - data_val.mean(dim=-2, keepdim=True)
    data_test = data_test - data_test.mean(dim=-2, keepdim=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", type=int, default=100)
    args = parser.parse_args()
    run(args)