from mimetypes import init
from networkx import center
import lean
from lean.openmm import OverdampedLangevinIntegrator
from lean.schedules import MeanFieldSinRBFSchedule
import openmm as mm
import numpy as np
import torch
import math

class CenteredNormal(torch.nn.Module):
    def __init__(self, log_sigma, particles, dimension=3):
        super().__init__()
        self.log_sigma = log_sigma
        self.particles = particles
        self.dimension = dimension
    
    def sample(self, shape=None):
        if shape is None:
            shape = (self.particles, self.dimension)
        x = torch.randn(shape) * self.log_sigma.exp()
        x = x - x.mean(dim=-2, keepdims=True)
        return x
    
    def force(self):
        DoF = self.dimension * (self.particles - 1)
        normalizing_constant = -0.5 * DoF * math.log(2*math.pi) - 0.5 * self.log_sigma
        force = mm.CustomExternalForce("b + k*((x/sigma)^2+(y/sigma)^2+(z/sigma)^2)")
        force.addGlobalParameter("k", -0.5)
        force.addGlobalParameter("sigma", self.log_sigma.exp().item())
        force.addGlobalParameter("b", normalizing_constant.item())
        return force
        
    def log_prob(self, x):
        N = x.shape[-2]
        D = x.shape[-1]
        DoF = D * (N - 1)
        normalizing_constant = -0.5 * DoF * math.log(2*math.pi) - 0.5 * self.log_sigma
        log_prob = normalizing_constant - 0.5 * (x / self.log_sigma.exp()) ** 2
        return log_prob
    
def dw_force(k2=-2.0, k4=0.45, d0=4.0):
    force = mm.CustomBondForce("k2*(r-d0)^2+k4*(r-d0)^4")
    force.addPerBondParameter("k2")
    force.addPerBondParameter("k4")
    force.addPerBondParameter("d0")
    return force

def dw2(batch_size=1):
    system = mm.System()
    force = dw_force()
    for idx in range(batch_size):
        system.addParticle(1.0)
        system.addParticle(1.0)
        force.addBond(2*idx, 2*idx+1, [-2.0, 0.45, 4.0])
    system.addForce(force)
    return system
        
class Policy(torch.nn.Module):
    def __init__(
        self,
        system: mm.System,
        integrator: mm.Integrator,
    ):
        super().__init__()
        self.centered_normal = CenteredNormal(torch.tensor(0.0), 2)
        self.system = system
        self._get_original()
        self._add_unbrella()
        self.schedules = torch.nn.ModuleDict(
            {
                "k2": MeanFieldSinRBFSchedule(10),
                "k4": MeanFieldSinRBFSchedule(10),
                "k_gaussian": MeanFieldSinRBFSchedule(10),
            }
        )
        self.integrator = integrator
        self.context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
        
        
    def _get_original(self):
        force = self.system.getForce(0)
        original_parameters = []
        for idx in range(force.getNumBonds()):
            original_parameters.append(force.getBondParameters(idx))
        self._original_parameters = original_parameters
        
    def _add_unbrella(self):
        force = self.centered_normal.force()
        self.system.addForce(force)
        
    def restore(self):
        force = self.system.getForce(0)
        for idx, parameters in enumerate(self._original_parameters):
            force.setBondParameters(idx, *parameters)
        force.updateParametersInContext(self.context)
        force = self.system.getForce(1)
        force.setGlobalParameterDefaultValue(0, 0.0)
        
    def sample(self):
        sampled = {}
        log_p = 0.0
        for key, schedule in self.schedules.items():
            gamma, coefficient = schedule.gamma, schedule.coefficient
            _gamma = gamma.sample()
            _coefficient = coefficient.sample()
            sampled[key] = (_gamma, _coefficient)
            log_p = log_p + gamma.log_prob(_gamma).sum() + coefficient.log_prob(_coefficient).sum()
        return sampled, log_p
            
    def get_coefficient(self, time, sampled):
        coefficients = {}
        for key, schedule in self.schedules.items():
            gamma, coefficient = sampled[key]
            lamb = schedule(time, gamma, coefficient)
            coefficients[key] = lamb
        return coefficients
    
    def perturb(self, coefficients):
        force = self.system.getForce(0)
        for idx in range(force.getNumBonds()):
            parameters = [coefficients["k2"], coefficients["k4"], 4.0]
            _, __, original_parameters = self._original_parameters[idx]
            parameters = [original_parameters[0] * parameters[0], original_parameters[1] * parameters[1], 4.0]
            force.setBondParameters(idx, 2*idx, 2*idx+1, parameters)
        force.updateParametersInContext(self.context)
        force = self.system.getForce(1)
        force.setGlobalParameterDefaultValue(0, 1-coefficients["k_gaussian"])
        force.updateParametersInContext(self.context)
    
    def trial(self):
        sampled, log_p = self.sample()
        self.integrator.ratio = 0.0
        initial_position = np.random.randn(2, 3)
        self.context.setPositions(initial_position)
        self.restore()
        for step in range(100):
            time = torch.tensor(step / 100.0)
            coefficients = self.get_coefficient(time, sampled)
            self.perturb(coefficients)
            self.integrator.step(100)
        self.restore()
        state = self.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy()._value
        positions = state.getPositions(asNumpy=True)._value
        # distance = np.linalg.norm(positions[0] - positions[1])
        initial_energy = self.centered_normal.log_prob(initial_position).sum()
        log_w = -torch.tensor(energy) + initial_energy + self.integrator.ratio
        # print(energy, initial_energy.item(), self.integrator.ratio)
        if np.isnan(log_w):
            log_w = torch.tensor(0.0)
        return log_w, log_p
        
        
def run():
    BATCH_SIZE = 10
    integrator = OverdampedLangevinIntegrator(100.0, 100.0, 1e-2)
    system = dw2()
    policy = Policy(system, integrator)
    
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-2,
    )
    
    counter = 0
    log_w, log_p = [], []
    for _ in range(10000000):
        _log_w, _log_p = policy.trial()
        log_w.append(_log_w)
        log_p.append(_log_p)
        counter += 1
        
        if counter % BATCH_SIZE == 0:
            log_w = torch.stack(log_w)
            log_p = torch.stack(log_p)
            ess = 1 / (log_w.softmax(0) ** 2).sum()
            loss = -(log_w.exp() * log_p).mean()
            loss.backward()
            log_w, log_p = [], []
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
            print(loss.item(), ess.item())
        optimizer.step()
    
    # for _ in range(100000):
    #     optimizer.zero_grad()
    #     log_w, log_p = policy.trial()
    #     loss = -log_w.exp() * log_p
    #     loss.backward()
    #     print(log_w.exp().mean().item())
    #     optimizer.step()
    
    # distances = []
    # for _ in range(100):
    #     _, __, distance = policy.trial()
    #     distances.append(distance)
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.kdeplot(distances)
    # plt.savefig("distance.png")
        
    
    

if __name__ == '__main__':
    run()