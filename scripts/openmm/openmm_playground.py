from networkx import center
import lean
from lean.openmm import OverdampedLangevinIntegrator
import openmm as mm
import numpy as np
from scripts.dw2.run import coefficient
import torch
import math
        
def dw_force(k2=-2.0, k4=0.45, d0=4.0):
    force = mm.CustomBondForce("k2*(r-d0)^2+k4*(r-d0)^4")
    force.addPerBondParameter("k2")
    force.addPerBondParameter("k4")
    force.addPerBondParameter("d0")
    # force.addGlobalParameter("k2", k2)
    # force.addGlobalParameter("k4", k4)
    # force.addGlobalParameter("d0", d0)
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

def perturb(system, context, k2, k4):
    force = system.getForce(0)
    for idx in range(force.getNumBonds()):
        force.setBondParameters(idx, 2*idx, 2*idx+1, [k2, k4, 4.0])
    force.updateParametersInContext(context)

def run():
    integrator = OverdampedLangevinIntegrator(1, 1.0, 0.01)
    system = dw2()
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    position = np.random.randn(2, 3)
    context.setPositions(position)
    
    # perturb(system, context, 1000000, 20000000)
    perturb(system, context, 50.0, 20.0)
    print(system.getForce(0).getBondParameters(0))
    integrator.step(100)
    
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole) 
    
    print(energy)
    
    

    

if __name__ == '__main__':
    run()