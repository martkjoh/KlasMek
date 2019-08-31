import numpy as np
from numpy import e, pi, sin, cos, exp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as anim
import shutil, os


m1 = l1 = m2 = l2 = 1
g = 10.
dt = 0.002
# Number of pendulums
N = 2

# b[n][m][l] is the nth doublependulum, mth mass, lth derivative w.r.t t 
bs = np.zeros((2, 2, 3))
bs[0][0][0] = pi 
bs[1][0][0] = pi 
bs[0][1][0] = pi + 0.001
bs[1][1][0] = pi + 0.0011

# One step of Eulers method
def updateState(bs):
    global m1, m2, g, dt
    for b in bs:
        b[0][2] = (-(m2 * l2 * b[1][2] * cos(b[0][0] - b[1][0]) + m2 * l2 * b[1][1]**2 * sin(b[0][0] - b[1][0]) + (m1 + m2) * g * sin(b[0][0]) ) / ((m1 + m2) * l1))
        b[1][2] = - (l1 * b[0][2] * cos(b[0][0] - b[1][0]) - l1 * b[0][1]**2 * sin(b[0][0] - b[1][0]) + g * sin(b[1][0])) / l2
        for j in range(2):
            b[j][1] += b[j][2] * dt
            b[j][0] += b[j][1] * dt
    return bs

# Returns pos. of origo, pendulum 1 and 2 in cartesian coordinates
def getXY(b):
    global l1, l2
    x1 = l1 * sin(b[0][0])
    y1 = - l1 * cos(b[0][0])
    x2 = x1 + l2 * sin(b[1][0])
    y2 = y1 - l2 * cos(b[1][0])
    return ([0, x1, x2], [0, y1, y2])


fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
lines = [ax.plot([], [], "o-") for i in range(N)]

def animate(n):
    global bs

    # Several timesteps increas prec. w/o need for high framerat / slow motion
    for i in range(10):
        bs = updateState(bs)

    for i in range(N):
        lines[i][0].set_data(*getXY(bs[i]))
    return lines

a = anim(fig, animate,  frames = 1000)
if os.path.isfile("dp_frames"):
    shutil.rmtree("dp_frames")

a.save("dp.html", fps = 24)
