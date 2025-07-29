#!/usr/bin/env python
"""
Generate app.pdf figure showing applications of entanglement distillation.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import entropy
from numpy import log2, exp
from math import isclose

# Set up plotting style
plt.style.use(['science'])
mpl.rc('font', family='serif')
sns.set_palette("muted")


def calc_p_succ(n, k, f):
    """Calculate success probability."""
    return (f ** n - 4 ** (-n)) / (1 - 4 ** (-n)) + ((1 - f ** n) / (1 - 4 ** (-n))) * (
                (2 ** (n - k) * 4 ** k) / 4 ** (n))


def calc_numerator(n, k, f):
    """Calculate numerator for fidelity calculation."""
    return (f ** n - 4 ** (-n)) / (1 - 4 ** (-n)) + ((1 - f ** n) / (1 - 4 ** (-n))) * (2 ** (n - k) / 4 ** (n))


def calc_new_f_root(n, k, f):
    """Calculate new fidelity root."""
    return (calc_numerator(n, k, f) / calc_p_succ(n, k, f)) ** (1 / k)


def calc_prob_f_after_distill(n, k, f):
    """Calculate probability and fidelity after distillation."""
    return calc_p_succ(n, k, f), calc_new_f_root(n, k, f)


def calc_f_after_swap(f):
    """Calculate fidelity after swap."""
    return f ** 2 + ((1 - f) ** 2) / 3


def h(p):
    """Binary entropy function."""
    if p == 1 or p == 0:
        return 0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def DEJMPS(F):
    """DEJMPS protocol calculation."""
    prob = F ** 2 + 2 * F * (1 - F) / 3 + 5 * (1 - F) ** 2 / 9
    return prob, (F ** 2 + ((1 - F) ** 2 / 9)) / prob


def calc_step(overhead, f, n, k):
    """Calculate step in protocol."""
    p, f = calc_prob_f_after_distill(n, k, f)
    f = calc_f_after_swap(f)
    return 2 * overhead * (n / (k * p)), f


def get_SKF(f):
    """Get secret key fraction."""
    x = (4 * f - 1) / 3
    qber = (1 + x) / 2
    return max(0, 1 - 2 * h(qber))


def calc_first_step(L, N, f, n, k):
    """Calculate first step with distance constraint."""
    psucc = exp(-L / (22 * 2 ** N+1))
    p, f = calc_prob_f_after_distill(n, k, f)
    overhead = n / (k * p * psucc)
    f = calc_f_after_swap(f)
    return 2 * overhead, f


def calc_first_step_DEJMPS(L, N, f, n):
    """Calculate first step for DEJMPS protocol."""
    psucc = exp(-L / (22 * 2 ** (N)))
    ptot = 1
    for i in range(n):
        p, f = DEJMPS(f)
        ptot *= p
    overhead = (2 ** n) / (ptot * psucc)
    f = calc_f_after_swap(f)
    return 2 * overhead, f,


def calc_step_DEJMPS(overhead_init, f, n):
    """Calculate step for DEJMPS protocol."""
    ptot = 1
    for i in range(n):
        p, f = DEJMPS(f)
        ptot *= p
    overhead = (2 ** n) / (ptot)
    f = calc_f_after_swap(f)
    return overhead_init * 2 * overhead, f


def calc_all(L, f_init, N, nks):
    """Calculate all steps of protocol."""
    assert N == len(nks)
    overhead, f = calc_first_step(L, N, f_init, nks[0][0], nks[0][1])
    for i in range(N - 1):
        overhead, f = calc_step(overhead, f, nks[i + 1][0], nks[i + 1][1])
    return overhead, f, get_SKF(f) / overhead


def calc_all_DEJMPS(L, f, N, nks):
    """Calculate all steps for DEJMPS protocol."""
    overhead, f = calc_first_step_DEJMPS(L, N, f, nks[0])
    for i in range(N - 1):
        overhead, f = calc_step_DEJMPS(overhead, f, nks[i + 1])
    return overhead, f, get_SKF(f) / overhead


def iter_all_DEJMPS_N(L, N, f_in):
    """Iterate over all DEJMPS protocols for given N."""
    poss_ns = range(0, 4)
    max_SKR = 0
    max_prot = None
    
    for ns2 in product(poss_ns, repeat=N - 1):
        for n1 in range(1, 11):
            ns = [n1] + list(ns2)
            overhead, f, SKR = calc_all_DEJMPS(L, f_in, N, ns)
            if SKR > max_SKR:
                max_SKR = SKR
                max_prot = ns
    return max_SKR, max_prot


def iter_all_random_protocols(L, N, f_in):
    """Iterate over all random protocols."""
    max_SKR = 0
    for n in range(4, 41):
        for k in range(1, n):
            init_ns = [(n, k)]
            poss_ns = [(n, n - 1), (n, n - 2)]
            for add_ns in product(poss_ns, repeat=N-1):
                add_ns = list(add_ns)
                ns = init_ns + add_ns
                overhead, f, SKR = calc_all(L, f_in, N, ns)
                if SKR > max_SKR:
                    max_SKR = SKR
                    opt_ns = ns
    return max_SKR, opt_ns


def iter_all_random_protocols_restricted(L, N, f_in):
    """Iterate over restricted set of random protocols."""
    max_SKR = 0
    for n in range(4, 13):
        for k in range(1, n):
            init_ns = [(n, k)]
            poss_ns = [(n, n - 1), (n, n - 2)]
            for add_ns in product(poss_ns, repeat=N-1):
                add_ns = list(add_ns)
                ns = init_ns + add_ns
                overhead, f, SKR = calc_all(L, f_in, N, ns)
                if SKR > max_SKR:
                    max_SKR = SKR
                    opt_ns = ns
    return max_SKR, opt_ns


def calc_hashing(f, L, N):
    """Calculate hashing rate."""
    ps = [f, (1 - f) / 3, (1 - f) / 3, (1 - f) / 3]
    rate = max(1 - entropy(ps, base=2), 0)
    psucc = exp(-L / (22 * 2 ** N))
    return rate * psucc / (2 ** N)


def calc_rate_overhead_pattison(eps, N, L):
    """Calculate rate for Pattison et al. method."""
    ov = get_pattison_overhead(eps)
    p = exp(-L / (2 ** N * 22))
    return p / (((ov) * 2 ** N))


def get_pattison_overhead(eps):
    """Get overhead values from Pattison et al."""
    if isclose(eps, 0.0035, abs_tol=1e-8):
        return 2.95
    elif isclose(eps, 0.0125, abs_tol=1e-8):
        return 5.2
    elif isclose(eps, 0.052, abs_tol=1e-8):
        return 12.99
    elif isclose(eps, 0.102, abs_tol=1e-8):
        return 27.16
    elif isclose(eps, 0.152, abs_tol=1e-8):
        return 67.32
    else:
        print(eps)
        assert 1 == 0


def plot_rate_as_func(L, f_in, ax, legend=False):
    """Plot rate as function of number of levels."""
    Ns = range(3, 8)
    
    skrs_dejmps = []
    skrs_random_restricted = []
    skrs_random = []
    skrs_hashing = []
    skrs_pattison = []
    
    for N in Ns:
        SKR_dejmps, prot = iter_all_DEJMPS_N(L, N, f_in)
        skrs_dejmps.append(SKR_dejmps)
        
        SKR, _ = iter_all_random_protocols_restricted(L, N, f_in)
        skrs_random_restricted.append(SKR)
        
        SKR, _ = iter_all_random_protocols(L, N, f_in)
        skrs_random.append(SKR)
        
        SKR = calc_rate_overhead_pattison(1-f_in, N, L)
        skrs_pattison.append(SKR)
        
        SKR = calc_hashing(f_in, L, N)
        skrs_hashing.append(SKR)
    
    ax.semilogy(Ns, skrs_random_restricted, '--')
    ax.semilogy(Ns, skrs_random, label='This work' if legend else None, color='C0')
    ax.semilogy(Ns, skrs_dejmps, linestyle='-', label="DEJMPS" if legend else None, c='C1')
    ax.semilogy(Ns, skrs_pattison, linestyle='-', label="Pattison et al." if legend else None, c='C2')
    ax.semilogy(Ns, skrs_hashing, label='Hashing' if legend else None, c='C3')


def iter_all_N_fixed_telescope(L, N, f_in):
    """Iterate over fixed telescope protocol."""
    ns = [(93, 68)] + (N - 1) * [(40, 39)]
    overhead, f, SKR = calc_all(L, f_in, N, ns)
    if f < 1 - 1e-9:
        return 0
    else:
        return 1 / overhead


def main():
    # Create figure with mosaic layout
    L = 1000
    fig, axes = plt.subplot_mosaic([
        ['.', 'A', 'A', 'A', 'A', '.'],
        ['.', '.', '.', '.', '.', '.'],
        ['B', 'B', 'C', 'C', 'D', 'D'],
    ], figsize=(4.3, 4), height_ratios=[0.44, 0.12, 0.44])
    
    # Part A: Distance vs Overhead plot
    Ls = np.linspace(1, 1000, 200)
    Ns = range(1, 10)
    
    hashing = []
    random = []
    pattison = []
    
    eps = 0.35 / 100
    f = 1 - eps
    
    for L in Ls:
        hashing_L = [calc_hashing(f, L, N) for N in Ns]
        hashing.append(1/max(hashing_L))
        
        random_L = [iter_all_N_fixed_telescope(L, N, f) for N in Ns]
        random.append(1/max(random_L))
        
        pattison_L = [calc_rate_overhead_pattison(1-f, N, L) for N in Ns]
        pattison.append(1/max(pattison_L))
    
    axes['A'].semilogy(Ls, random, c='C0')
    axes['A'].semilogy(Ls, pattison, c='C2')
    axes['A'].semilogy(Ls, hashing, c='C3')
    axes['A'].set_xlabel("Distance (km)")
    axes['A'].set_ylabel(r"Overhead $\mathcal{O}$")
    axes['A'].grid(True)
    
    # Parts B, C, D: Rate plots for different error rates
    plot_rate_as_func(1000, 1-15.2/100, axes['B'], legend=True)
    plot_rate_as_func(1000, 1-5.2/100, axes['C'])
    plot_rate_as_func(1000, 1-0.35/100, axes['D'])
    
    # Add labels and titles
    fig.text(0.0, 0.43, s=r'\bf b)', fontsize=12)
    fig.text(0.15, 0.86, s=r'\bf a)', fontsize=12)
    axes['B'].set_ylabel("Secret bits/Bell pair", fontsize=8)
    
    axes['C'].set_xlabel(r"Number of levels")
    axes['B'].set_title(r'$\varepsilon_0=0.152$')
    axes['C'].set_title(r'$\varepsilon_0=0.052$')
    axes['D'].set_title(r'$\varepsilon_0=0.0035$')
    
    # Configure axes
    for ax in [axes['B'], axes['C'], axes['D']]: 
        ax.set_xlim((2.5, 7.5))
        ax.set_ylim((7e-6, 1.2e-2))
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    axes['C'].yaxis.set_ticklabels([])
    axes['D'].yaxis.set_ticklabels([])
    
    # Add legend
    fig.legend(ncol=4, bbox_to_anchor=(0.5, 0.0), loc='upper center', fontsize=8, frameon=True)
    
    # Save figure
    fig.savefig('figs/app.pdf', bbox_inches='tight')
    print("Generated figs/app.pdf")


if __name__ == "__main__":
    main()