import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from Wave2D import Wave2D_Neumann


wave = Wave2D_Neumann()

N = 30
Nt = 150
CFL = 1 / np.sqrt(2)
mx = my = 2

plotdata = wave(N, Nt+1, cfl=CFL, c=1.0, mx=mx, my=my, store_data=10)

xij, yij = wave.xij, wave.yij

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

frames = []
for n, val in plotdata.items():
    frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2)
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
ani.save("report/neumannwave.gif", writer='pillow', fps=5, dpi=150)



