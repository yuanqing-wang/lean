from simtk.openmm.app import Simulation, StateDataReporter, DCDReporter
from simtk.openmm import LangevinIntegrator, Platform
from simtk.unit import kelvin, picosecond, femtoseconds
import mdtraj as md
import numpy as np
from openmmtools import testsystems

# Load alanine dipeptide system
test_system = testsystems.AlanineDipeptideVacuum(constraints=None)
system = test_system.system
positions = test_system.positions

# Define the integrator
temperature = 300 * kelvin
friction = 1 / picosecond
time_step = 2 * femtoseconds
integrator = LangevinIntegrator(temperature, friction, time_step)

# Create the simulation
# platform = Platform.getPlatformByName('Reference')  # Use GPU if available
simulation = Simulation(test_system.topology, system, integrator)

# Set initial positions
simulation.context.setPositions(positions)

# Minimize energy
print("Minimizing energy...")
simulation.minimizeEnergy()

# Set up reporters for output
simulation.reporters.append(md.reporters.HDF5Reporter('trajectory.h5', 1000))

# Run MD simulation
print("Running simulation...")
simulation.step(100_000_000)  # Run 100 ps (100,000 steps with 2 fs timestep)

print("Simulation complete!")

