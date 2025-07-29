#!/usr/bin/env python
"""
Generate evolution-small.pdf figure showing finite-depth evolution with noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.special

# Set up plotting style
plt.style.use(['science'])
mpl.rc('font', family='serif')
sns.set_palette("muted")


def log_binom(n, k):
    """Calculate log of binomial coefficient."""
    ret = scipy.special.gammaln(n+1) - scipy.special.gammaln(k+1) - scipy.special.gammaln(n-k+1)
    return np.where((k>n)|(k<0), -np.inf, ret)


def p0_f(n, noise):
    """Initial probability distribution."""
    j = np.arange(n+1)
    return np.exp(log_binom(n, j) + j*np.log(noise) + (n-j) * np.log(1-noise))


def build_transition_noisy(n, f0, f1, f2):
    """Build noisy transition matrix."""
    ret = np.zeros((n+1, n+1))
    for w in range(n+1):
        # Case w' = w+2
        if w + 2 <= n:
            ret[w+2, w] = (3 * (1 - f0) * (n - w) * (n - w - 1)) / (5 * n * (n - 1))
        # Case w' = w+1
        if w + 1 <= n:
            ret[w+1, w] = ((2 * (1 - f0) * (n - w) * (n - w - 1)) + (6 * (1 - f1) * w * (n - w))) / (5 * n * (n - 1))
        ret[w, w] = ((5 * f0 * (n - w) * (n - w - 1)) + (4 * (1 - f1) * w * (n - w)) + (3 * (1 - f2) * w * (w - 1))) / (5 * n * (n - 1))
        if w - 1 >= 0:
            ret[w-1, w] = ((10 * f1 * w * (n - w)) + (2 * (1 - f2) * w * (w - 1))) / (5 * n * (n - 1))
        if w - 2 >= 0:
            ret[w-2, w] = (f2 * w * (w - 1)) / (n * (n - 1))
    
    return ret


def get_probs(dist, n, k):
    """Calculate probabilities for given distribution."""
    m = n - k
    ws = np.arange(n+1)
    numerator = np.exp(scipy.special.logsumexp(-ws * np.log(3) + log_binom(m, ws) - log_binom(n, ws) + np.log(dist)))    
    denom = []
    for w in ws:
        j = np.arange(max(0, w-m), min(w, k)+1)
        denom.append(scipy.special.logsumexp(log_binom(k, j) + log_binom(m, w-j) - log_binom(n, w) + (j-w)*np.log(3)))
    denom = np.array(denom) + np.log(dist)
    denom = np.exp(scipy.special.logsumexp(denom))
    return numerator, denom, numerator/denom


def main():
    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    
    # Parameters
    n = 30
    p0 = p0_f(n, 0.02)
    
    # Create logarithmically spaced noise levels
    noise_levels = np.logspace(-7, -3, 6)
    
    # Create a colormap that goes from blue (low noise) to red (high noise)
    cmap = plt.cm.plasma
    norm = mpl.colors.LogNorm(vmin=1e-7, vmax=1e-3)
    
    nits2 = np.linspace(n, 5*n*np.log2(n), 100).astype(int)
    k = int(0.25*n)
    
    # Add the noiseless case first
    f0, f1 = 1, 0
    f2 = f1
    T = build_transition_noisy(n, f0, f1, f2)
    probs = np.array([get_probs(np.linalg.matrix_power(T, nits2[j]) @ p0, n=n, k=k) for j in range(len(nits2))])
    ax.semilogy(nits2/n, 1-(probs[:,2])**(1/k), 'k--', label='Noiseless')
    
    # Plot for each noise level
    for lamb in noise_levels:
        f0, f1 = (1-lamb)**2 + lamb**2/15, lamb/15 * (2 - 16/15 * lamb)
        f2 = f1
        T = build_transition_noisy(n, f0, f1, f2)
        probs = np.array([get_probs(np.linalg.matrix_power(T, nits2[j]) @ p0, n=n, k=k) for j in range(len(nits2))])
        line = ax.semilogy(nits2/n, 1-(probs[:,2])**(1/k), color=cmap(norm(lamb)))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'Noise strength $\lambda$')
    
    # Configure plot
    ax.grid()
    ax.grid(which='minor', ls='--', alpha=0.5)
    ax.set_ylabel(r'$\bar{\varepsilon}$')
    ax.set_xlabel(r'$G/n$')
    fig.tight_layout()
    
    # Save figure
    fig.savefig('figs/evolution-small.pdf', bbox_inches='tight')
    print("Generated figs/evolution-small.pdf")


if __name__ == "__main__":
    main()