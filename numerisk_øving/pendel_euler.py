import numpy as np
from numpy import pi, cos, sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

m = 1
l = 1
g = 9.81
T = 10

font = {'family' : 'serif', 
        'weight' : 'normal', 
        'size'   : 22}


plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc("lines", lw=2)
plt.rc('font', **font)
plt.rc("legend", fontsize = 12)

def Energy(theta):
    return 1 / 2 * m * (l * theta[:,1])**2 + m * g * l * (1 - cos(theta[:,0]))

def f(theta):
    return np.array([theta[1], -g / l * theta[0]])

def Euler(theta, dt, j):
    theta[j + 1] = theta[j] + f(theta[j]) * dt

def EulerCromer(theta, dt, j):
    theta[j + 1][1] = theta[j][1] - g / l * theta[j][0] * dt
    theta[j + 1][0] = theta[j][0] + theta[j + 1][1] * dt
    
def RK4(theta, dt, j):
    k1 = f(theta[j]) * dt
    k2 = f(theta[j] + k1 / 2) * dt
    k3 = f(theta[j] + k2 / 2) * dt
    k4 = f(theta[j] + k3) * dt
    theta[j + 1]  = theta[j] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def runMethod(N, dt0, Method, ax1, ax2):
    labels = [''] * (N + 1)

    for i in range(N):
        dt = dt0 * (i + 1)**2
        n = int(T / dt)

        # theta[i][j] is the j'th derivative of theta at
        theta = np.zeros((n, 2))
        t = np.linspace(0, T, n)

        theta[0][0] = 0.2

        # Very scary use of pass-by-reference
        for j in range(n - 1):
            Method(theta, dt, j)


        # plotting theta and E against t
        labels[i] = "$\Delta t = " + "{:.3f}".format(dt) + "$"
        ax1[0].plot(t, theta[:,0], color = cm.viridis(i / N))
        ax1[0].set_xlabel("$t$")
        ax1[0].set_ylabel("$\\theta$")

        ax1[1].plot(t, Energy(theta), color = cm.coolwarm(i / N))
        ax1[1].set_xlabel("$t$")
        ax1[1].set_ylabel("$E$")
        ax1[1].set_ylim(0, 0.8)

        # plotting phase space
        ax2.plot(theta[:, 0], theta[:, 1], color = cm.viridis(i / N))
        ax2.set_xlabel("$\\theta$")
        ax2.set_ylabel("$\dot \\theta$")


    x = np.linspace(0, T, 100)
    ax1[0].plot(x, theta[0][0] * cos(sqrt(g / l) * x), "r--")
    labels[-1] = "analytical solution"
    ax1[0].legend(labels, loc = "upper left")
    ax1[1].legend(labels[:-1])
    ax2.legend(labels[:-1])

    return ax1, ax2

N = 3
dts = [0.005, 0.05, 0.03]
methods = [Euler, EulerCromer, RK4]

for i in range(3):
    fig1 = plt.figure(figsize = (10, 10))
    ax1 = fig1.subplots(2)
    fig2 = plt.figure(figsize = (10, 10))
    ax2 = fig2.subplots(1)
    runMethod(N, dts[i], methods[i], ax1, ax2)

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig("numerisk_øving/fig" + str(i))
    fig2.savefig("numerisk_øving/fig" + str(i + 3))    
