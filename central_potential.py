import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

k = 1
m = 1

def V(r):
    return - k / r

def VCent(r, l):
    return l**2  / (2 * m * r**2)

def VEff(r, l):
    return V(r) + VCent(r, l)

r = np.linspace(0, 10, 1000)

fig, ax = plt.subplots(1)
ax.set_ylim(-1, 1)
ax.plot(r, V(r), "k--")

for i in range(1, 10):
    l = i / 2
    ax.plot(r, VEff(r, l), color = cm.viridis(i / 10))

ax.plot(r, r*0, "b--")

plt.show()
fig.savefig("Orbit-potentials.png")