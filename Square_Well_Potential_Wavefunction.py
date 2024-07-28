import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
import numba
from numba import jit
from scipy.linalg import eigh_tridiagonal
import scienceplots
plt.style.use(['science', 'notebook'])

# def GetPotential(x):
#     V = []
#     for i in range(len(x)):
#         V.append(0)
#     V[0] = 1e10
#     V[len(x)-1] = 1e10
#     return V

L = 1
Nx = 301
Nt = 100000
dx = L / (Nx - 1)
dt = 1e-7
x = np.linspace(0, 1, Nx)
psi0 = np.sqrt(2)*np.sin(np.pi*x)
mu, sigma = 1/2, 3/20
V = -1e4*np.exp(-(x-mu)**2/(2*sigma**2))
# V = -1e4*np.sin(3 * np.pi * x)
psi = np.zeros([Nt, Nx])
psi[0] = psi0

@numba.jit("c16[:,:](c16[:,:])", nopython=True, nogil=True)
def compute_psi(psi):
    for t in range(0, Nt-1):
        for i in range(1, Nx - 1):
            psi[t+1][i] = psi[t][i] + 1j/2 * dt/dx**2 * (psi[t][i+1] - 2*psi[t][i] + psi[t][i-1]) - 1j*dt*V[i]*psi[t][i]
    
    normal = np.sum(np.absolute(psi[t+1])**2)*dx
    for i in range (1, Nx-1):
        psi[t+1][i] = psi[t+1][i]/normal

    return psi

psi = compute_psi(psi.astype(complex))

def PositionOperator(psi):
    position = np.zeros([Nt, Nx])
    for t in range(len(psi)):
        for i in range(len(psi[0])):
            position[t][i] = psi[t][i] * x[i]
    return position
position = PositionOperator(psi)

def animate(i):
    ln[0].set_data(x, np.absolute(psi[100*i])**2)
    ln_position[0].set_data(x, np.absolute(position[100*i])**2)
    return ln
fig, ax = plt.subplots(1, 1)
ln = plt.plot([], [], 'r-', lw=2, markersize=8, color='red')
ln_position = plt.plot([], [], 'r-', lw=2, markersize=8, color='blue')
ax.set_ylim(-1, 20)
ax.set_xlim(0, 1)
ax.set_title("Quantum Wavefunction")
plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, frames=1000, interval=30)
ani.save('/Users/aaronkang/Downloads/hand_v1_L1.123c38c9cfe4-2d46-434b-995c-4f7cd8b7ab01/square_well.gif', writer='pillow', fps=50, dpi=100)

plt.show()




