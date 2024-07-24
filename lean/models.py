import jax
import jax.numpy as jnp
from flax import linen as nn
import tensorflow_probability.substrates.jax.distributions as tfd
from .layers import EGNNLayer

class EGNNModel(nn.Module):
    """A stack of EGNN layers. """
    hidden_features: int
    depth: int
    activation: nn.Module = nn.silu
    
    def setup(self):
        self.layers = [
            EGNNLayer(
                self.hidden_features,
                self.hidden_features,
            ) for _ in range(self.depth)
        ]
        
    def __call__(self, h, x):
        for layer in self.layers:
            h, x = layer(h, x)
            h = self.activation(h)
        return h, x
    
class ConditionalVelocity(nn.Module):
    """Sample and measure velocity conditioned on the positions. """
    hidden_features: int
    depth: int
    
    def setup(self):
        self.egnn_mu = EGNNModel(self.hidden_features, self.depth)
        self.egnn_log_sigma = EGNNModel(self.hidden_features, self.depth)
        
    def __call__(self, h, x, key):
        _, mu = self.egnn_mu(h, x)
        _, log_sigma = self.egnn_log_sigma(h, x)
        mu = mu - x
        mu = mu - jnp.mean(mu, axis=-2, keepdims=True)
        v = jax.random.normal(key, mu.shape) * jnp.exp(log_sigma) + mu
        return v
    
    def log_prob(self, h, x, v):
        _, mu = self.egnn_mu(h, x)
        _, log_sigma = self.egnn_log_sigma(h, x)
        mu = mu - x
        mu = mu - jnp.mean(mu, axis=-2, keepdims=True)
        log_prob = tfd.Normal(mu, jnp.exp(log_sigma)).log_prob(v)
        return log_prob
        
        
        
    
        
        
        