import numpy as np
from numpy import pi, cos, sqrt
import matplotlib.pyplot as plt
from matplotlib import cm


font = {'family' : 'serif', 
    'weight' : 'normal', 
    'size'   : 16}
    
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc("lines", lw = 1.5)
plt.rc('font', **font)
plt.rc("legend", fontsize = 10)

m = 1
l = 1
g = 9.81
theta0 = 0.2

def Energy(theta):
    return 1 / 2 * m * (l * theta[:, 1])**2 + m * g * l * (1 - cos(theta[:, 0]))

# Uses a subarray of the usual theta, only need one index
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

def simulate(T, dt, method):
    n = int(T / dt)

    # theta[i][j] is the j'th derivative of theta at time i
    theta = np.zeros((n, 2))
    t = np.linspace(0, T, n)
    theta[0][0] = theta0
    # Very scary use of pass-by-reference
    for j in range(n - 1):
        method(theta, dt, j)
    
    return (t, theta)

def plotTheta(t, theta, ax, color, label):
    ax.plot(t, theta[:,0], color = color, label = label)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\theta$")

def plotE(t, E, ax, color, label):
    ax.plot(t, E, color = color, label = label)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$E$")

    ymax = min(0.8, np.max(E * 1.6))
    ymax = max(ymax, ax.get_ylim()[1])
    ax.set_ylim(0,  ymax)

def plotPhase(theta, ax, color, label):
    ax.plot(theta[:, 0], theta[:, 1], color = color, label = label)
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$\dot \\theta$")

def runMethod(N, T, dt0, method, ax, axPhase = 0):
    x = np.linspace(0, T, 50)
    ax[0].plot(x, theta0 * cos(sqrt(g / l) * x), "kx", label = "analytical solution")

    for i in range(N):
        dt = dt0 * (i + 1)**2
        t, theta = simulate(T, dt, method)

        color = cm.viridis((i + 1 / 2) / N)
        label = "$\Delta t = " + "{:.3f}".format(dt) + "$"
        plotTheta(t, theta, ax[0], color, label)
        plotE(t, Energy(theta), ax[1], color, label)

        ax[0].legend(loc = "upper left")
        ax[1].legend()

        if axPhase:
            plotPhase(theta, axPhase, color, label)
            axPhase.legend()


def plot(N, T, dt0, methods, names, fname, phasePlot = False):
    n = len(methods)

    if phasePlot:
        # Fig for phase-space plot
        figPhase = plt.figure(figsize = (4 * n, 3))
        axPhase = figPhase.subplots(1, n)
        figPhase.subplots_adjust(wspace = 0.45)
        for i in range(n):
            axPhase[i].set_title(names[i])
    else:
        axPhase = [0] * n

    for i in range(n):
        # Fig for angle, energy as func of t
        fig = plt.figure(figsize = (12, 3))
        ax = fig.subplots(1, 2)
        fig.suptitle(names[i])
        fig.subplots_adjust(wspace = 0.3, bottom = 0.2)

        runMethod(N, T, dt0, methods[i], ax, axPhase[i])

        fig.savefig(fname + "_" + str(i) + ".pdf")

    if phasePlot:
        figPhase.savefig(fname + "_phase.pdf") 

path = "numerisk_øving/bilder/"
methods = [Euler, EulerCromer, RK4]
names = ["Euler", "Euler-Cromer", "Runge-Kutta 4"]

plot(3, 10, 0.005, [methods[0]], [names[0]], path + "euler")
plot(1, 6, 0.05, methods, names, path + "all_methods", phasePlot = True)
plot(3, 10, 0.04, methods[1::], names[1::], path + "ECvsRK4", phasePlot = True)
