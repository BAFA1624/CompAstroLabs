import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit


def read_data(filename):
    x, y = [], []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            vals = [np.float64(s) for s in line.strip().split(',')]
            x.append(vals[0])
            y.append(vals[1])
        x = np.array(x)
        y = np.array(y)
    return x, y


def p1(x): return (2/np.pi)*(np.sin(x)**2)
def p2(x): return (0.375)*(1+x**2)
def p3(x): return (0.375)*(1+(np.cos(x)**2)) * np.sin(x)
def p4(x): return np.exp(-x)


mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.sans-serif"] = "Times New Roman"
mpl.rcParams["font.family"] = "sans-serif"


def mu(theta):
    return np.cos(theta)


def p(mu):
    return (3/8) * (1+mu**2)


theta_vals = np.linspace(-np.pi, np.pi, 10000)
mu_vals = mu(theta_vals)
p_vals = p(np.linspace(-1, 1, 10000))


fig1 = plt.figure(figsize=(8, 8))

fig1.suptitle(
    r"Expected Probability Distribution, $-\pi \leq \theta \leq \pi$", fontsize=16)

ax1 = fig1.add_subplot(111)
ax2 = ax1.twiny()
ax3 = ax2.twinx()

ax1.plot(np.linspace(-1., 1., len(p_vals)), p_vals, 'k-', label=r"$P(\mu)$")
ax1.set_ylim(0, None)
ax1.set_xlim(-1, 1)
ax1.set_xlabel(r"$\mu$", fontsize=14)
ax1.set_ylabel(r"$P(\mu)=\frac{3}{8}(1+\mu^2)$", fontsize=14)
ax1.tick_params(direction='in')

ax2.set_xlabel(r"$\theta$", fontsize=14)
ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_xticklabels(
    [r"$-\pi$", r"$-\frac{\pi}{2}$", r"0", r"$\frac{\pi}{2}$", r"$\pi$"])
ax2.tick_params(direction='in')
ax3.plot(theta_vals, mu_vals, 'r--', label=r"$\mu(\theta)$")
ax3.set_xlim(min(theta_vals), max(theta_vals))
ax3.set_ylabel(r"$\mu(\theta)=\cos(\theta)$", fontsize=14)
ax3.tick_params(direction='in')

fig1.legend()

fig1.savefig("expected_probability")


yticks = np.arange(0.25, 0.875, 0.125)

x, y = read_data('1a_rejection_method.csv')

fig2 = plt.figure(figsize=(12, 12))
ax1 = fig2.add_subplot(111)

ax1.plot(x, y, 'ko', ls='none', markersize=1, label='Distribution')
ax1.plot(x, p2(x), 'r--', label='Model')
ax1.set_xlim(min(x), max(x))
ax1.set_ylim(None, max(y))
ax1.set_title('Rejection Method')
ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'$P(\mu)$')
ax1.tick_params(direction='in')
ax1.set_yticks(yticks, [r'$\frac{1}{4}$', r'$\frac{3}{8}$',
                        r'$\frac{1}{2}$', r'$\frac{5}{8}$', r'$\frac{3}{4}$'])

fig = plt.figure(figsize=(20, 50))

fig.suptitle("Monte-Carlo methods, N=1000000 & M=1000")

gs = gridspec.GridSpec(2, 1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

ax1.plot(x, y, 'ko', ls='none', markersize=1, label='Distribution')
ax1.plot(x, p2(x), 'r--', label='Model')
ax1.set_xlim(min(x), max(x))
ax1.set_ylim(None, max(y))
ax1.set_title('Rejection Method')
ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'$P(\mu)$')
ax1.tick_params(direction='in')
ax1.set_yticks(yticks)
ax1.set_yticklabels([r'$\frac{1}{4}$', r'$\frac{3}{8}$',
                     r'$\frac{1}{2}$', r'$\frac{5}{8}$', r'$\frac{3}{4}$'])

x, y = read_data('1a_importance_sampled.csv')

ax2.plot(x, y, 'ko', ls='none', markersize=1, label="Distribution")
ax2.plot(x, p2(x), 'r--', label="Model")
ax2.set_xlim(min(x), max(x))
ax2.set_ylim(None, max(y))
ax2.set_title('Importance Sampling')
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel(r'$P(\mu)$')
ax2.tick_params(direction='in')
ax2.set_yticks(yticks)
ax2.set_yticklabels([r'$\frac{1}{4}$', r'$\frac{3}{8}$',
                    r'$\frac{1}{2}$', r'$\frac{5}{8}$', r'$\frac{3}{4}$'])

ax1.legend()
ax2.legend()


x, y = read_data('1b_norm_intensity.csv')

fig3 = plt.figure(figsize=(12, 12))
ax1 = fig3.add_subplot(111)

ax1.plot(np.arccos(x), y, 'ko', ls='none')


plt.show()
