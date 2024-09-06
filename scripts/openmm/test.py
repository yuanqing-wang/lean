import lean
from lean.openmm import OverdampedLangevinIntegrator
import openmm as mm
import numpy as np
import jax
jax.clear_caches()


def dw2():
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = mm.CustomBondForce(
        "-2*(r-4.0)^2+0.45*(r-4.0)^4"
    )
    force.addBond(0, 1, [])
    system.addForce(force)
    return system
    
    
def dw1000():
    system = mm.System()
    for i in range(10000):
        system.addParticle(1.0)
    force = mm.CustomBondForce(
        "-2*(r-4.0)^2+0.45*(r-4.0)^4"
    )
    for i in range(9999):
        force.addBond(i, i+1, [])
    system.addForce(force)
    return system

def run():
    integrator = OverdampedLangevinIntegrator(1, 100, 0.01)
    # system = dw2()
    system = dw1000()
    positions = np.random.rand(10000, 3)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    
    integrator.step(1)
    
    import time
    time0 = time.time()
    for _ in range(1000):
        integrator.step(10)
    time1 = time.time()
    print("Time: ", time1-time0)
    
    # get position
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)
    print(positions)
    
    
    
if __name__ == '__main__':
    run()