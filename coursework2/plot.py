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
def p4(x, a=1, b=1): return a * np.exp(-b*x)


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


fig1 = plt.figure(figsize=(7, 5))

# fig1.suptitle(
#     r"Expected Probability Distribution, $-\pi \leq \theta \leq \pi$", fontsize=16)

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

x1, y1 = read_data('1b_norm_intensity_tau_0.100000.csv')
x2, y2 = read_data('1b_norm_intensity_tau_1.000000.csv')
x3, y3 = read_data('1b_norm_intensity_tau_10.000000.csv')
x4, y4 = read_data('1b_norm_intensity_tau_15.000000.csv')
x5, y5 = read_data('1b_norm_intensity_tau_20.000000.csv')

fig3 = plt.figure(figsize=(12, 12))
ax1 = fig3.add_subplot(111)

ax1.set_title("Isotropic Scattering")
ax1.plot(np.arccos(x2), y2, c='k', marker='o',
         markersize=3, ls='none', label=r'$\tau = 1$')
ax1.plot(np.arccos(x3), y3, c='k', marker='x',
         markersize=3, ls='none', label=r'$\tau = 10$')
ax1.plot(np.arccos(x5), y5, c='k', marker='D',
         markersize=3, ls='none', label=r'$\tau = 20$')

ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"Normalized Intensity")
ax1.legend()
ax1.tick_params(direction='in')
fig3.tight_layout()

fig4 = plt.figure(figsize=(8, 5))
ax1 = fig4.add_subplot(111)

x, y = read_data('rejection_error.csv')
ax1.loglog(x, y, c='k', marker='x', markersize=5,
           ls='none', label='Rejection Method')

x, y = read_data('importance_error.csv')
ax1.loglog(x, y, c='k', marker='o', markersize=5,
           ls='none', label='Importance Sampling')

ax1.set_xlabel(r"Number of samples, $N$")
ax1.set_ylabel(r"MSE($N$)")
ax1.tick_params(which='both', direction='in')
fig4.tight_layout()

fig4.legend()

x1, y1 = read_data('1c_norm_intensity_tau_0.100000.csv')
x2, y2 = read_data('1c_norm_intensity_tau_1.000000.csv')
x3, y3 = read_data('1c_norm_intensity_tau_10.000000.csv')
x4, y4 = read_data('1c_norm_intensity_tau_15.000000.csv')
x5, y5 = read_data('1c_norm_intensity_tau_20.000000.csv')

fig5 = plt.figure(figsize=(12, 12))
ax1 = fig5.add_subplot(111)
ax1.axhline(0, c='k', linewidth=0.8)
ax1.plot(np.arccos(x2), y2, c='k', marker='o',
         markersize=3, ls='none', label=r'$\tau = 1$')
ax1.plot(np.arccos(x3), y3, c='k', marker='x',
         markersize=3, ls='none', label=r'$\tau = 10$')
ax1.plot(np.arccos(x5), y5, c='k', marker='D',
         markersize=3, ls='none', label=r'$\tau = 20$')
xtmp = np.linspace(0, 1, 1000)

ax1.set_title("Thomson Scattering")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"Normalized Intensity")
ax1.tick_params(direction='in')
ax1.legend()

fig6 = plt.figure(figsize=(12, 12))
ax1 = fig6.add_subplot(111)

x1, y1 = read_data('1b_norm_intensity_tau_1.000000.csv')
x2, y2 = read_data('1c_norm_intensity_tau_1.000000.csv')

ax1.plot(np.arccos(x1), y1, c='k', marker='o', markersize=5,
         ls='none', label='Isotropic Scattering')
ax1.plot(np.arccos(x2), y2, c='k', marker='^', markersize=5,
         ls='none', label='Thomson Scattering')

ax1.set_title(r'$\tau = 1$')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'Normalized Intensity')
ax1.tick_params(direction='in')
ax1.legend()

fig7 = plt.figure(figsize=(12, 12))
ax1 = fig7.add_subplot(111)

x1, y1 = read_data('1b_norm_intensity_tau_10.000000.csv')
x2, y2 = read_data('1c_norm_intensity_tau_10.000000.csv')

ax1.plot(np.arccos(x1), y1, c='k', marker='o', markersize=5,
         ls='none', label='Isotropic Scattering')
ax1.plot(np.arccos(x2), y2, c='k', marker='^', markersize=5,
         ls='none', label='Thomson Scattering')

ax1.set_title(r'$\tau = 10$')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'Normalized Intensity')
ax1.tick_params(direction='in')
ax1.legend()

fig8 = plt.figure(figsize=(12, 12))
ax1 = fig8.add_subplot(111)

x1, y1 = read_data('1b_norm_intensity_tau_20.000000.csv')
x2, y2 = read_data('1c_norm_intensity_tau_20.000000.csv')

ax1.plot(np.arccos(x1), y1, c='k', marker='o', markersize=5,
         ls='none', label='Isotropic Scattering')
ax1.plot(np.arccos(x2), y2, c='k', marker='^', markersize=5,
         ls='none', label='Thomson Scattering')

ax1.set_title(r'$\tau = 20$')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'Normalized Intensity')
ax1.tick_params(direction='in')
ax1.legend()
fig5.tight_layout()

plt.show()
