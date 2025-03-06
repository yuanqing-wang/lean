from openmmtools import testsystems
import mdtraj
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


def evaluate_aldp(traj):
    x_np = traj.xyz

    # Compute Ramachandran plot angles
    aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    sampled_traj = mdtraj.Trajectory(x_np.reshape(-1, 22, 3), topology)
    psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
    phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
    not_nan = np.logical_not(is_nan)
    psi = psi[not_nan]
    phi = phi[not_nan]

    nbins = 200
    hgen_phi, _ = np.histogram(phi, nbins, range=[-np.pi, np.pi], density=True)
    hgen_psi, _ = np.histogram(psi, nbins, range=[-np.pi, np.pi], density=True)
    

    # Plot phi and psi
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    x = np.linspace(-np.pi, np.pi, nbins)
    ax[0].plot(x, hgen_phi, linewidth=3)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].set_xlabel('$\phi$', fontsize=24)
    ax[1].plot(x, hgen_psi, linewidth=3)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].set_xlabel('$\psi$', fontsize=24)
    plt.savefig('phi_psi.png', dpi=300)
    plt.close()

    # Ramachandran plot
    plt.figure(figsize=(10, 10))
    plt.hist2d(phi, psi, bins=64, norm=matplotlib.colors.LogNorm(),
                range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$\phi$', fontsize=24)
    plt.ylabel('$\psi$', fontsize=24)
    plt.savefig('ramachandran.png',
                dpi=300)
    plt.close()
    
if __name__ == '__main__':
    traj = mdtraj.load('trajectory.h5')
    evaluate_aldp(traj)