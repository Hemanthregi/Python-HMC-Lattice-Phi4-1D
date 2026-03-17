# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:13:48 2026

@author: heman
"""


import numpy as np
import matplotlib.pyplot as plt


# Leapfrog integrator: symplectic update preserving reversibility and volume
# Performs half-step momentum update, full position updates, then final half-step
def leapfrog(phi, pi, epsilon, L, gradU):
    assert L > 0, "L must be greater than 0" #Failsafe for L being \leq 0 
    # First half-step update for momentum
    pi_half = pi - 0.5 * epsilon * gradU(phi)

    for step in range(L):

        # Full step update for position
        phi_next = phi + epsilon * pi_half
        

        # Full momentum update except at final step
        if step != L - 1:
            pi_next = pi_half - epsilon * gradU(phi_next)

            phi = phi_next
            pi_half = pi_next

    # Final half-step update for momentum
    pi_next = pi_half - 0.5 * epsilon * gradU(phi_next)

    return phi_next, -pi_next # Negate momentum for detailed balance (time-reversibility)

# Hamiltonian Monte Carlo
# One HMC update:
# 1. Sample Gaussian momentum
# 2. Propose new state via Hamiltonian dynamics (leapfrog)
# 3. Accept/reject using Metropolis criterion based on Hamiltonian change
def HMC(phi, epsilon, L, U, gradU):
    assert L > 0, "L must be greater than 0" #Failsafe for L being \leq 0 

    # Sample momentum from a Gaussian distribution
    pi = np.random.normal(size=len(phi)) #pi should be a vector of length of phi

    # Simulate Hamiltonian dynamics using leapfrog
    phi_new, pi_new = leapfrog(phi, pi, epsilon, L, gradU)

    # Compute Hamiltonian values
    H_current  = U(phi)     + 0.5 * np.sum(pi**2) 
    H_proposed = U(phi_new) + 0.5 * np.sum(pi_new**2)

    # Log acceptance ratio for the Metropolis test
    log_r = H_current - H_proposed

    accepted = False

    # Accept proposal with probability min(1, exp(log_r))
    if np.log(np.random.rand()) < log_r:
       phi = phi_new
       accepted = True

    return phi, accepted

def integrate(f, U, gradU, n_sites, n_samples, epsilon, L):
    """
Monte Carlo expectation value using Hamiltonian Monte Carlo (HMC)
for a 1D lattice phi^4 scalar field theory.

The algorithm samples field configurations phi from a probability
distribution proportional to exp(-S[phi]), where S is the lattice action.
These samples are used to estimate expectation values of observables.

Parameters
----------
f : function
    Observable evaluated on the field configuration phi.

U : function
    Action S[phi], defining the sampling distribution.

gradU : function
    Gradient of the action with respect to phi.

n_sites : int
    Number of lattice sites.

n_samples : int
    Number of HMC samples.

epsilon : float
    Leapfrog step size.

L : int
    Number of leapfrog steps per trajectory.

Returns
-------
result : float
    Estimated expectation value of f(phi).

acceptance_rate : float
    Fraction of accepted proposals.

samples : ndarray
    Collected field configurations.
"""

    phi = np.zeros(n_sites)
    accepted_count = 0
    values = []
    samples = []

    for _ in range(n_samples):

        # Perform one HMC step
        phi, accepted = HMC(phi, epsilon, L, U, gradU)

        if accepted:
            accepted_count += 1

        # Use the current state of the chain in the estimator
        values.append(f(phi))
        samples.append(phi)

    values = np.array(values)
    samples = np.array(samples)

    # Monte Carlo estimate of the expectation value
    result = np.mean(values)

    acceptance_rate = accepted_count / n_samples

    return result, acceptance_rate, samples

# Lattice phi^4 action: U(phi) ≡ S[phi] (Euclidean action)
# kinetic term (nearest-neighbor coupling) + mass term + quartic interaction
def U(phi):
    phi = np.array(phi)
    
    phi_plus  = np.roll(phi, -1)
    
    return np.sum(
        0.5 * (phi_plus - phi)**2   # gradient term
        + 0.5 * phi**2              # m^2 = 1
        + (1/24) * phi**4             # Lambda = 1
    )

# Gradient of the action:
# discrete Laplacian + mass term + nonlinear phi^3 interaction
def gradU(phi):
    phi = np.array(phi)
    
    phi_plus  = np.roll(phi, -1)
    phi_minus = np.roll(phi, 1)
    
    laplacian = phi_plus + phi_minus - 2*phi
    
    return -laplacian + phi + (1/6)*phi**3

def f(phi):
    return np.mean(phi**2)

result, acc, samples = integrate(
    f, U, gradU,
    50,        # n_sites
    200000,      # n_samples  NOTE: storing all samples is memory-intensive; acceptable for small lattices
    0.3,      # epsilon
    20         # L
)

print("Expectation:", result)
print("Acceptance rate:", acc)

# Computes spatial two-point function C(r) = <phi(x) phi(x+r)>
def correlator(samples, n_sites):
    
    max_r = n_sites // 2 
    # Say N=10, then for a cyclic lattice distance r = N -r, then r_max = N/2
                         
    
    C = []
    
    for r in range(max_r):
        
        values = []
        
        for phi in samples:
            val = np.mean(phi * np.roll(phi, -r)) #phi_i*phi_{i+r}
            values.append(val)
        
        C.append(np.mean(values))
    
    return np.array(C)

C = correlator(samples, n_sites = 50)

#############################################################
# MASS EXTRACTION (Effective Mass Method)
#############################################################

# Effective mass from correlator ratio:
# m_eff(r) = log(C(r)/C(r+1))
# Derived from C(r) ~ exp(-m r)

def effective_mass(C):
    """
    Compute effective mass from correlator.

    Parameters:
        C : array
            Correlator values C(r)

    Returns:
        m_eff : array
            Effective mass as function of r
    """
    C= np.abs(C)  # Take absolute value to avoid log issues from noise-induced sign flips
               # (not strictly physical, but stabilizes estimator)
    return np.log(C[:-1] / C[1:]) 


# Compute effective mass
m_eff = effective_mass(C)


#############################################################
#Visualization
#############################################################

#1. Correlation Plot
plt.plot(C[:6])
plt.yscale('log')
plt.xlabel("r")
plt.ylabel("C(r)")
plt.title(r"Spatial Correlator $C(r) = \langle \phi(x)\phi(x+r) \rangle$")
plt.savefig("hmc_phi4_correlation.png", dpi=300)
plt.show()

#2. Histogram
phi_all = samples.flatten()

plt.figure()
plt.hist(phi_all,
         bins=100,
         density=True,
         alpha=0.7)

plt.xlabel("phi")
plt.ylabel("Probability density")
plt.title(r"Field Distribution in $\phi^4$ Theory")
plt.savefig("hmc_phi4_hist.png", dpi=300)
plt.show()

#3. Markov Chain Trace

obs = [np.mean(phi**2) for phi in samples]

plt.figure()
plt.plot(obs[:1000])
plt.xlabel("Iteration")
plt.ylabel(r"$\langle \phi^2 \rangle$")
plt.title(r"HMC Trace of $\langle \phi^2 \rangle$")
plt.savefig("hmc_phi4_trace.png",dpi=300)
plt.show()

#4. Effective Mass

plt.figure()
plt.plot(m_eff, marker='o')
plt.xlabel("r")
plt.ylabel("m_eff(r)")
plt.title("Effective Mass (Intermediate region estimates mass)")
plt.grid()
plt.savefig("hmc_phi4_mass.png",dpi=300)
plt.show()


#############################################################
# Extract physical mass from plateau region
#############################################################

# IMPORTANT:
# You MUST choose this window based on the plot
# Typical starting guess:
r_min, r_max = 5,9

# Mean value = mass
m = np.mean(m_eff[r_min:r_max])

print("Extracted mass (m_phys):", m)