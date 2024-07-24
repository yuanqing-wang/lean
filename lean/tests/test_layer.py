import jax
import jax.numpy as jnp

def test_egnn_layer():
    from lean.layers import EGNNLayer
    layer = EGNNLayer(hidden_features=5, out_features=6)
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
    h = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
    params = layer.init(jax.random.PRNGKey(0), h, x)
    y = layer.apply(params, h, x)
    