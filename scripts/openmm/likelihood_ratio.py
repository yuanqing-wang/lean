from math import log
import torch

N = 2

def target(x):
    return x + torch.randn_like(x) * 1e-2

def run():
    mu = torch.nn.Parameter(torch.randn(N, N))
    log_sigma = torch.nn.Parameter(torch.randn(N, N))
    optimizer = torch.optim.Adam([mu, log_sigma], lr=1e-2)
    
    count = 0
    for _ in range(10000000000):
        optimizer.zero_grad()
        x = torch.randn(N)
        y = target(x)
        distribution = torch.distributions.Normal(mu, log_sigma.exp())
        theta = distribution.sample()
        log_prob = distribution.log_prob(theta).sum()
        with torch.no_grad():
            y_hat = x @ theta
        loss = (y - y_hat).pow(2).mean()
        print(mu.flatten(), log_sigma.exp().flatten())
        loss = loss * log_prob
        loss.backward()
        
        count += 1
        if count % 10 == 0:
            optimizer.step()
            count = 0
        
if __name__ == "__main__":
    run()