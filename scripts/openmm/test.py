from mimetypes import init
from networkx import center
import lean
from lean.openmm import OverdampedLangevinIntegrator
from lean.schedules import MeanFieldSinRBFSchedule
import openmm as mm
import numpy as np
import torch
import math
import copy
from multiprocessing import Pool

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

def perturb(original_parameters, coefficients, system, context):
    force = system.getForce(0)
    for idx in range(force.getNumBonds()):
        parameters = [coefficients["k2"], coefficients["k4"], 4.0]
        _, __, _original_parameters = original_parameters[idx]
        parameters = [_original_parameters[0] * parameters[0], _original_parameters[1] * parameters[1], 4.0]
        force.setBondParameters(idx, 2*idx, 2*idx+1, parameters)
    force.updateParametersInContext(context)
    force = system.getForce(1)
    force.setGlobalParameterDefaultValue(0, 1-coefficients["k_gaussian"])
    force.updateParametersInContext(context)
    
def add_umbrella(system):
    force = mm.CustomExternalForce("k*((x/sigma)^2+(y/sigma)^2+(z/sigma)^2)")
    force.addGlobalParameter("k", -0.5)
    force.addGlobalParameter("sigma", 1.0)
    system.addForce(force)
    return force
    
def _get_original(system):
    force = system.getForce(0)
    original_parameters = []
    for idx in range(force.getNumBonds()):
        original_parameters.append(force.getBondParameters(idx))
    return original_parameters

def restore(system, original_parameters):
    force = system.getForce(0)
    for idx in range(force.getNumBonds()):
        force.setBondParameters(idx, *original_parameters[idx])
    force = system.getForce(1)
    force.setGlobalParameterDefaultValue(0, 1.0)
        
class Policy(torch.nn.Module):
    def __init__(
        self,
        system: mm.System,
        batch_size: int = 10,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.centered_normal = CenteredNormal(torch.tensor(0.0), 2)
        self.system = system
        self.schedules = torch.nn.ModuleDict(
            {
                "k2": MeanFieldSinRBFSchedule(10),
                "k4": MeanFieldSinRBFSchedule(10),
                "k_gaussian": MeanFieldSinRBFSchedule(10),
            }
        )
                        
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
            
    @staticmethod
    def integration(positions, coefficients, system, original_parameters):
        add_umbrella(system)
        integrator = OverdampedLangevinIntegrator(100.0, 100.0, 1e-2)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
        integrator.ratio = 0.0
        context.setPositions(positions)
        for coefficient in coefficients:
            perturb(original_parameters, coefficient, system, context)
            integrator.step(100)
        # restore(system, original_parameters)
        energy = context.getState(getEnergy=True).getPotentialEnergy()._value
        ratio = integrator.ratio
        return energy, ratio
        
    
    def trial(self):
        samples, log_p_batch = zip(*[self.sample() for _ in range(self.batch_size)])
        coefficients = [[self.get_coefficient(time, sampled) for time in torch.linspace(0, 1, 10)] for sampled in samples]
        positions = [self.centered_normal.sample().detach().numpy() for _ in range(self.batch_size)]
        systems = [copy.deepcopy(self.system) for _ in range(self.batch_size)]
        energies, ratios = zip(*Pool().starmap(self.integration, zip(positions, coefficients, systems, [_get_original(system) for system in systems])))
        energies_initial = [self.centered_normal.log_prob(torch.tensor(position)).sum() for position in positions]
        log_w_batch = [-energy + ratio + energy_initial for energy, ratio, energy_initial in zip(energies, ratios, energies_initial)]
        return torch.stack(log_w_batch), torch.stack(log_p_batch)
            

def run():
    BATCH_SIZE = 10
    system = dw2()
    policy = Policy(system, BATCH_SIZE)
    
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-2,
    )
    
    for _ in range(10000000000):
        optimizer.zero_grad()
        log_w, log_p = policy.trial()
        loss = -(log_w.exp() * log_p).mean()
        loss.backward()
        optimizer.step()
        ess = 1 / (log_w.softmax(-1) ** 2).sum()
        print(ess)
    
        
    
    

if __name__ == '__main__':
    run()