import numpy as np
from numpy import sin, less_equal, greater, ones_like
from matplotlib import pyplot as plt

kZZq_e2 = 1

def sigma(theta, E = 2):
    a = 1 / 4 * (kZZq_e2 / (2 * E))**2 * (1 / sin(theta / 2))** 4
    b = np.ones_like(a)
    c = less_equal(a, b)
    d = greater(a, b)
    return a * c + (2 - 1 / a) * d

theta = np.linspace(-np.pi, np.pi, 100)

fig, ax = plt.subplots(1, subplot_kw = dict(polar = True))
ax.plot(theta, sigma(theta), label = "$\\sigma(\\Theta)$")
ax.plot(theta, np.ones_like(theta), label = "$\\sigma = 1$")
ax.plot(theta, np.zeros_like(theta), label = "$\sigma = 2$")

ax.set_rlim(-1, 3)
ax.grid(False)
ax.set_thetagrids([0, 180])
ax.set_rticks([])
plt.legend()
plt.show()
fig.savefig("rutherford_scattering.png")