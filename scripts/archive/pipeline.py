import numpy as np
import matplotlib.pyplot as plt
import pyqg
from pyqg import diagnostic_tools as tools
from pyqg import Smagorinsky
from ring_forcing import RingForcing

kf, dkf, Af = 8, 1, 1
k_in, k_out = kf - dkf, kf + dkf
C_nu = 0.1
rek = 0.1
N_grid = 256
forcing = RingForcing(k_in_forc=k_in, k_out_forc=k_out, mag_noise_forc=Af)
viscocity = Smagorinsky(constant=C_nu)

m = pyqg.BTModel(
    L = 2*np.pi, nx=N_grid,
    tmax=100, dt=0.01, taveint=0.5,
    beta=0., rek=rek, rd=None,
    ntd=4,
    q_parameterization = forcing,
    uv_parameterization = viscocity,
)

# define a quick function for plotting and visualize the initial condition
def plot_q(m, qmax=1):
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(m.x,m.y,m.q.squeeze(), cmap='RdBu_r')
    # pc.set_clim([-qmax, qmax])
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([0, 2*np.pi])
    ax.set_aspect(1)
    plt.colorbar(pc)
    plt.title('Time = %g' % m.t)
    plt.savefig("data/q.png")
    plt.close()
    
    energy = m.get_diagnostic('KEspec')
    enstrophy = m.get_diagnostic('Ensspec')

    kr, energy_iso = tools.calc_ispec(m,energy.squeeze())
    _, enstrophy_iso = tools.calc_ispec(m,enstrophy.squeeze())

    ks = np.array([3.,80])
    es = 5*ks**-4
    plt.loglog(kr,energy_iso)
    plt.loglog(ks,es,'k--')
    plt.text(2.5,.0001,r'$k^{-4}$',fontsize=20)
    plt.ylim(1.e-10,1.e0)
    plt.xlabel('wavenumber')
    plt.title('Energy Spectrum')
    plt.savefig("data/spec.png")
    plt.close()

for _ in m.run_with_snapshots(tsnapstart=0, tsnapint=1):
    plot_q(m)