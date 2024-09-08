from networkx import center
import lean
from lean.openmm import OverdampedLangevinIntegrator
from lean.schedules import MeanFieldSinRBFSchedule
import openmm as mm
import numpy as np
from scripts.dw2.run import coefficient
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
        
def dw_force(k2=-2.0, k4=0.45, d0=4.0):
    force = mm.CustomBondForce("k2*(r-d0)^2+k4*(r-d0)^4")
    force.addGlobalParameter("k2", k2)
    force.addGlobalParameter("k4", k4)
    force.addGlobalParameter("d0", d0)
    return force

def dw2(batch_size=1):
    system = mm.System()
    force = dw_force()
    for idx in range(batch_size):
        system.addParticle(1.0)
        system.addParticle(1.0)
        force.addBond(2*idx, 2*idx+1, [])
    system.addForce(force)
    return system

def system_with_centered_normal(batch_size, log_sigma):
    system = dw2(batch_size=batch_size)
    centered_normal = CenteredNormal(log_sigma, batch_size)
    force = centered_normal.force()
    system.addForce(force)
    return system

def perturb(system, context, k2, k4, k_gaussian):
    force = system.getForce(0)
    force.setGlobalParameterDefaultValue(0, k2)
    force.setGlobalParameterDefaultValue(1, k4)
    force.updateParametersInContext(context)
    force = system.getForce(1)
    force.setGlobalParameterDefaultValue(0, k_gaussian)
    force.updateParametersInContext(context)
    # update force 
    
    
def restore(system, context, k2=-2.0, k4=0.45):
    force = system.getForce(0)
    force.setGlobalParameterDefaultValue(0, k2)
    force.setGlobalParameterDefaultValue(1, k4)
    force.updateParametersInContext(context)
    force = system.getForce(1)
    force.setGlobalParameterDefaultValue(0, 0.0)
    force.updateParametersInContext(context)
    
def run():
    BATCH_SIZE = 1

    integrator = OverdampedLangevinIntegrator(1, 1.0, 0.01)
    system = system_with_centered_normal(BATCH_SIZE, torch.tensor(0.0))
    
    schedule = MeanFieldSinRBFSchedule(10)
    optimizer = torch.optim.Adam(schedule.parameters(), lr=1e-2)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    
    for _ in range(10000000):
        optimizer.zero_grad()
        positions = np.random.rand(BATCH_SIZE*2, 3)
        context.setPositions(positions)
        perturb(system, context, 0.0, 0.0, 1.0)
        initial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)

        gamma, coefficient = schedule.gamma, schedule.coefficient
        _gamma, _coefficient = gamma.sample(), coefficient.sample()
        log_p = gamma.log_prob(_gamma).sum() + coefficient.log_prob(_coefficient).sum()
        
        for step in range(100):
            time = torch.tensor(step / 100.0)
            lamb = schedule(time, gamma=_gamma, coefficient=_coefficient)
            perturb(system, context, 1.0*lamb, 0.45*lamb, lamb)
            integrator.step(100)

        restore(system, context)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole) 
        # energy = energy - initial_energy + integrator.ratio
        
        integrator.ratio = 0.0
        loss = energy * log_p
        loss.backward()
        print(energy)
        optimizer.step()
    
    

    
    integrator.ratio = 0.0
    # run simulation
    

    

if __name__ == '__main__':
    run()