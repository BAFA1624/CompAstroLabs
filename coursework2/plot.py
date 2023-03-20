import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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


fig = plt.figure(figsize=(8,8))

fig.suptitle(r"Expected Probability Distribution, $-\pi \leq \theta \leq \pi$", fontsize=16)

ax1 = fig.add_subplot(111)
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
ax2.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"0", r"$\frac{\pi}{2}$", r"$\pi$"])
ax2.tick_params(direction='in')
ax3.plot(theta_vals, mu_vals, 'r--', label=r"$\mu(\theta)$")
ax3.set_xlim(min(theta_vals), max(theta_vals))
ax3.set_ylabel(r"$\mu(\theta)=\cos(\theta)$", fontsize=14)
ax3.tick_params(direction='in')

fig.legend()

fig.tight_layout()

fig.savefig("expected_probability")
plt.show()
