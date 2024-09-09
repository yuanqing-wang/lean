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

def system_with_centered_normal(batch_size, log_sigma):
    system = dw2(batch_size=batch_size)
    centered_normal = CenteredNormal(log_sigma, batch_size)
    force = centered_normal.force()
    system.addForce(force)
    return system

def perturb(system, context, k2, k4, k_gaussian):
    force = system.getForce(0)
    for idx in range(force.getNumBonds()):
        force.setBondParameters(idx, 2*idx, 2*idx+1, [k2, k4, 4.0])
    force.updateParametersInContext(context)
    force = system.getForce(1)
    force.setGlobalParameterDefaultValue(0, k_gaussian)
    force.updateParametersInContext(context)
    
def restore(system, context, k2=-2.0, k4=0.45):
    force = system.getForce(0)
    for idx in range(force.getNumBonds()):
        force.setBondParameters(idx, 2*idx, 2*idx+1, [k2, k4, 4.0])
    force.updateParametersInContext(context)
    force = system.getForce(1)
    force.setGlobalParameterDefaultValue(0, 0.0)
    force.updateParametersInContext(context)
    
def run():
    BATCH_SIZE = 1

    integrator = OverdampedLangevinIntegrator(1, 1.0, 0.01)
    system = system_with_centered_normal(BATCH_SIZE, torch.tensor(0.0))
    
    schedule2 = MeanFieldSinRBFSchedule(10)
    schedule4 = MeanFieldSinRBFSchedule(10)
    schedule_gaussian = MeanFieldSinRBFSchedule(10)
    optimizer = torch.optim.Adam(
        list(schedule2.parameters()) + list(schedule4.parameters()) + list(schedule_gaussian.parameters()), 
        lr=1e-2,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    
    for _ in range(10000000):
        optimizer.zero_grad()
        positions = np.random.rand(BATCH_SIZE*2, 3)
        context.setPositions(positions)
        initial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)

        gamma2, coefficient2 = schedule2.gamma, schedule2.coefficient
        gamma4, coefficient4 = schedule4.gamma, schedule4.coefficient
        gamma_gaussian, coefficient_gaussian = schedule_gaussian.gamma, schedule_gaussian.coefficient
        
        _gamma2 = gamma2.sample()
        _coefficient2 = coefficient2.sample()
        _gamma4 = gamma4.sample()
        _coefficient4 = coefficient4.sample()
        _gamma_gaussian = gamma_gaussian.sample()
        _coefficient_gaussian = coefficient_gaussian.sample()
        
        log_p = gamma2.log_prob(_gamma2).sum() + coefficient2.log_prob(_coefficient2).sum() + \
                gamma4.log_prob(_gamma4).sum() + coefficient4.log_prob(_coefficient4).sum() + \
                gamma_gaussian.log_prob(_gamma_gaussian).sum() + coefficient_gaussian.log_prob(_coefficient_gaussian).sum()
        
        for step in range(100):
            time = torch.tensor(step / 100.0)
            lamb2 = schedule2(time, _gamma2, _coefficient2).item()
            lamb4 = schedule4(time, _gamma4, _coefficient4).item()
            lamb_gaussian = schedule_gaussian(time, _gamma_gaussian, _coefficient4).item()

            perturb(system, context, 1.0*lamb2, 0.45*lamb4, 1-lamb_gaussian)
            
            integrator.step(100)

        restore(system, context)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole) 
        # energy = energy - initial_energy + integrator.ratio
        print(energy)
        if ~np.isnan(energy):
            loss = energy * log_p
            loss.backward()
            optimizer.step()
    
    

    
    integrator.ratio = 0.0
    # run simulation
    

    

if __name__ == '__main__':
    run()