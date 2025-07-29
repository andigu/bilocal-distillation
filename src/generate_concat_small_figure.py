#!/usr/bin/env python
"""
Generate concat-small.pdf figure showing expected overhead vs epsilon_0 for different levels.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.special
import scipy.optimize
import scipy.linalg
import functools as ft
import multiprocess as mp
from tqdm.auto import tqdm
import qiskit.quantum_info as qi

# Set up plotting style
plt.style.use(['science'])
mpl.rc('font', family='serif')
sns.set_palette("muted")

# Constants
TARGET_INFED = 1e-12


@ft.cache
def log_binom(n, k):
    """Calculate log of binomial coefficient."""
    return scipy.special.gammaln(n+1) - scipy.special.gammaln(k+1) - scipy.special.gammaln(n-k+1)


def with_qec(n, k, L, f):
    """Calculate fidelity with quantum error correction."""
    m = int(n-k)
    q = 0
    term2 = 0
    weight = 1
    l = 2
    while L > 0:
        num_err = min(3**weight * scipy.special.binom(n, weight), L)
        q += num_err*((1-f)/3)**weight * f**(n-weight)
        term2 += 2**(-m) * (((l-1)+(l+num_err-1))/2 * num_err) * ((1-f)/3)**weight * f**(n-weight)
        l += num_err
        L -= num_err
        weight += 1
    q += f**n
    numerator = max(q - term2, 0)
    denom = q + 2**(-m)*(l-1)*(1-q)
    return (numerator/denom)**(1/k), q


def no_qec(n, k, f):
    """Calculate fidelity without quantum error correction."""
    m = int(n-k)
    return (1/(2**(-m) * (f**(-n)-1) + 1))**(1/k), f**n


def rate_f0(params, f0=0.9, return_dat=False):
    """Rate function for depth 0."""
    n1, k1_frac, Lfrac = params
    n1 = int(np.round(n1))
    k1 = int(np.clip(np.round(k1_frac*n1), 1, n1-1))
    L = int(np.round(min(2**(n1-k1), 3e6) * Lfrac))
    f1, prob1 = with_qec(n1, k1, L, f0)
    if return_dat: 
        total_n = n1
        overhead = 1/(prob1) * total_n/k1
        return [(n1,),(k1,),(f1,),(prob1,),L,overhead]
    if f1 < 1-TARGET_INFED: 
        return np.inf
    else: 
        total_n = n1
        return -np.log(prob1) + np.log(total_n) - np.log(k1)


def rate_f1(params, f0=0.9, return_dat=False):
    """Rate function for depth 1."""
    n1, k1_frac, mult2, k2_frac, Lfrac = params
    n1 = int(np.round(n1))
    mult2 = int(np.round(mult2))
    k1 = int(np.clip(np.round(k1_frac*n1), 1, n1-1))
    L = int(np.round(min(2**(n1-k1), 3e6) * Lfrac))
    n2 = k1*mult2
    if n2 == 1: 
        return np.inf
    k2 = int(np.clip(np.round(n2*k2_frac), 1, n2-1))
    f1, prob1 = with_qec(n1, k1, L, f0)
    f2, prob2 = with_qec(n2, k2, 0, f1)
    if return_dat: 
        total_n = mult2*n1
        overhead = 1/(prob1*prob2) * total_n/k2
        return [(n1,n2),(k1,k2),(f1,f2),(prob1,prob2),L,overhead]
    if f2 < 1-TARGET_INFED: 
        return np.inf
    else: 
        total_n = mult2*n1
        return -np.log(prob1)-np.log(prob2) + np.log(mult2) + np.log(n1) - np.log(k2)


def rate_f2(params, f0=0.9, return_dat=False):
    """Rate function for depth 2."""
    n1, k1_frac, mult2, k2_frac, mult3, k3_frac, Lfrac = params
    mult2, mult3 = np.round(mult2), np.round(mult3)
    n1 = np.round(n1)
    k1 = np.clip(np.round(k1_frac*n1), 1, n1-1)
    L = np.round(min(2**(n1-k1), 3e6) * Lfrac)
    n2 = k1*mult2
    k2 = np.clip(np.round(n2*k2_frac), 1, n2-1)
    n3 = k2*mult3
    k3 = np.clip(np.round(n3*k3_frac), 1, n3-1)
    if n2 == 1 or n3 == 1: 
        return np.inf
    f1, prob1 = with_qec(n1, k1, L, f0)
    f2, prob2 = with_qec(n2, k2, 0, f1)
    f3, prob3 = with_qec(n3, k3, 0, f2)
    if return_dat: 
        total_n = mult2*mult3*n1
        overhead = 1/(prob1*prob2*prob3) * total_n/k3
        return [(n1,n2,n3),(k1,k2,k3),(f1,f2,f3),(prob1,prob2,prob3),L,overhead]
    if f3 < 1-TARGET_INFED: 
        return np.inf
    else: 
        total_n = mult2*mult3*n1
        ret = -np.log(prob1)-np.log(prob2)-np.log(prob3) + np.log(mult2*mult3*n1) - np.log(k3)
        return ret if ret >= 0 else np.inf


def get_rho(p=0.1):
    """Get density matrix for given error probability."""
    bell = np.array([1,0,0,1])/np.sqrt(2)
    states = [qi.Pauli(x + 'I').to_matrix() @ bell for x in ['I','X','Y','Z']]
    rho = (1-p)*np.outer(np.conj(states[0]), states[0]) + p/3 * np.sum([np.outer(np.conj(x), x) for x in states[1:]], axis=0)
    return rho


def rains(x, error_prob):
    """Calculate Rains bound."""
    probs = scipy.special.softmax(x[:4])
    theta = x[4:12]
    phi = x[12:20]
    states = np.stack([np.cos(theta), np.exp(1j*phi)*np.sin(theta)], axis=-1)
    state1 = states[:4]
    state2 = states[4:]
    sigma = np.sum([p*np.kron(np.outer(np.conj(x), x), np.outer(np.conj(y), y)) for (p, x, y) in zip(probs, state1, state2)], axis=0)
    rho = get_rho(error_prob)
    return np.real(np.trace(rho @ (scipy.linalg.logm(rho) - scipy.linalg.logm(sigma))))/np.log(2)


def get_popt(infed, depth=2, with_qec=True):
    """Get optimal parameters for given infidelity and depth."""
    func = None
    bounds = None
    integrality = None
    if depth == 2: 
        func = rate_f2
        bounds=[(4, 500),(0.,1),(1,30),(0.,1),(1,20),(0.,1),(0,1 if with_qec else 0)]
        integrality = [True, False, True, False, True, False, False]
    elif depth == 1:
        func = rate_f1
        bounds=[(4, 500),(0.,1),(1,30),(0.,1),(0,1 if with_qec else 0)]
        integrality = [True, False, True, False, False]
    else: 
        func = rate_f0
        bounds=[(4, 1000),(0.,1),(0,1 if with_qec else 0)]
        integrality = [True, False, False]
    ret = scipy.optimize.differential_evolution(func, bounds=bounds, args=(1-10**infed,), popsize=800, maxiter=10000, integrality=integrality)
    return func(ret.x, return_dat=True, f0=1-10**infed), infed


def main():
    # Generate infidelity values
    infeds = np.linspace(np.log10(0.0035), -0.8, 16)
    
    # Calculate Rains bound
    print("Calculating Rains bound...")
    rs = np.array([scipy.optimize.minimize(rains, x0=np.random.rand(20), args=(p,)).fun for p in tqdm(10**infeds)])
    
    # Calculate optimal parameters for different depths and QEC settings
    print("Calculating optimal parameters...")
    popts = [[[None,None,None], [None,None,None]] for _ in infeds]
    with mp.Pool(8) as p:
        for depth in [0,1,2]:
            for active in [0,1]:
                for ret in tqdm(p.imap_unordered(ft.partial(get_popt, depth=depth, with_qec=bool(active)), infeds), 
                               total=len(infeds), leave=False):
                    dat, infed = ret
                    popts[np.argwhere(infeds == infed).flatten()[0]][active][depth] = dat
    
    # Create figure
    plt.figure(figsize=(3.5,3.1))
    plt.ylabel(r'$\mathbb{E}[\mathcal{O}]-1$')
    plt.xlabel(r'$\varepsilon_0$')
    
    # Plot results for different depths
    plt.loglog(10**infeds, [x[0][0][-1]-1 for x in popts], c='C0', ls='--')
    plt.loglog(10**infeds, [x[1][0][-1]-1 for x in popts], c='C0', label=r'$L=0$')
    plt.loglog(10**infeds, [x[0][1][-1]-1 for x in popts], c='C1', ls='--')
    plt.loglog(10**infeds, [x[1][1][-1]-1 for x in popts], c='C1', label=r'$L=1$')
    plt.loglog(10**infeds, [x[0][2][-1]-1 for x in popts], c='C2', ls='--')
    plt.loglog(10**infeds, [x[1][2][-1]-1 for x in popts], c='C2', label=r'$L=2$')
    
    # Add comparison data
    plt.plot([0.0035, 0.0125, 0.052, 0.102, 0.152], np.array([2.95, 5.2, 12.99, 27.16, 67.32])-1, 
             label='Pattison et al.', c='grey', marker='*')
    plt.plot([0.0035, 0.0125, 0.052, 0.102, 0.152], np.array([1089, 1369, 5329, 22201, 142129])-1, 
             label='Lattice surgery', c='grey', marker='x')
    
    # Add lower bound
    plt.loglog(10**infeds, 1/rs-1, c='black', label='Lower bound')
    
    # Configure plot
    plt.grid()
    plt.grid(which='minor', ls='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=8, ncol=2, framealpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig("figs/concat-small.pdf", bbox_inches='tight')
    print("Generated figs/concat-small.pdf")


if __name__ == "__main__":
    main()