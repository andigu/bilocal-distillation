#!/usr/bin/env python
"""
Generate phase.pdf figure showing output fidelity and epsilon bar vs m/n ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set up plotting style
plt.style.use(['science'])
mpl.rc('font', family='serif')
sns.set_palette("muted")


def no_qec(n, k, f):
    """Calculate output fidelity without quantum error correction."""
    m = int(n - k)
    output_fidelity = (1 / (2**(-m) * (f**(-n) - 1) + 1))**(1/k)
    success_prob = f**n
    return output_fidelity, success_prob


def main():
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.6), sharex=True)
    
    # Plot for different values of n
    for n in [25, 50, 100, 500]:
        ks = np.arange(1, n+1)
        fids = np.array([no_qec(n, k, 0.8)[0] for k in ks])
        
        # Left subplot: Output fidelity
        axes[0].plot(1 - ks/n, fids**ks)
        
        # Right subplot: Epsilon bar
        axes[1].plot(1 - ks/n, 1 - fids, label=n)
    
    # Add vertical line at specific value
    tmp = np.round(-np.log2(0.8), 5)
    axes[0].axvline(tmp, c='black', ls='--')
    axes[1].axvline(tmp, c='black', ls='--')
    
    # Configure axes
    axes[1].legend(title=r'$n$', frameon=True)
    axes[0].set_xlabel(r'$m/n$')
    axes[1].set_xlabel(r'$m/n$')
    axes[0].set_ylabel(r'Output fidelity $\bar{f}^k$')
    axes[1].set_ylabel(r'$\bar{\varepsilon}$')
    axes[0].grid()
    axes[1].grid()
    
    # Save figure
    fig.tight_layout()
    fig.savefig('figs/phase.pdf', bbox_inches='tight')
    print("Generated figs/phase.pdf")


if __name__ == "__main__":
    main()