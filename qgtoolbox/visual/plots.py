import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cmocean.cm as cmo

from matplotlib.colors import LogNorm

YELLOW = '#ffff99'
GREEN = '#7fc87f'
BLUE = '#385eb1'

def plot_potential_vorticity(ds, axis, t_idx, cmap='cmo.curl', levels=50):
    x = ds.x.values
    y = ds.y.values
    q = ds.q.isel(time=t_idx)
    Q = np.abs(ds.q).max().values
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, q/Q, cmap=cmap, levels=scale)
    axis.set_title(f'Potential Vorticity $q$ | $Q = {Q:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_absolute_velocity(ds, axis, t_idx, cmap='cmo.speed', levels=50):
    x = ds.x.values
    y = ds.y.values
    V = np.abs(ds.v).max().values
    v = ds.v.isel(time=t_idx)
    scale = np.linspace(0.0, +1, levels)
    cf = axis.contourf(x, y, v/V, cmap=cmap, levels=scale)
    axis.set_title(f'Absolute Velocity $v$, | $V = {V:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_stream_function(ds, axis, t_idx, cmap='cmo.delta', levels=50):
    x = ds.x.values
    y = ds.y.values
    P = np.abs(ds.p).max().values
    p = ds.p.isel(time=t_idx)
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, p/P, cmap=cmap, levels=scale)
    axis.set_title(f'Stream Function $\\psi$ | $\\Psi = {P:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_energy_field(ds, axis, t_idx, cmap='cmo.dense'):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    beta = ds.attrs['model.beta']
    epsilon = A_f**2/k_f**2

    k_nu = 1/3 * n_grid
    k_b = (beta**3/epsilon)**(1/5)
    k_r = (n_grid * r_ek*beta**2/epsilon)**(1/4)
    
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    k_bx = k_b * (np.cos(theta))**(8/5)
    k_by = k_b * np.sin(theta) * (np.cos(theta))**(3/5)

    kx = ds.kx.values
    ky = ds.ky.values
    eh = ds.eh.isel(time=t_idx)
    Eh = np.abs(ds.eh).max().values
    norm = LogNorm(vmin=1e-6, vmax=1e1)
    mesh = axis.pcolormesh(kx, ky, eh/Eh, cmap=cmap, norm=norm)
    axis.set_title(f'Energy Spectrum $\\hat E(k_x, k_y)$ | $ \\hat E = {Eh:.1e}$')
    # axis.set_aspect('equal')
    axis.set_xlabel('Kx')
    axis.set_ylabel('Ky')

    axis.add_patch(patches.Circle((0, 0), radius=k_nu, edgecolor='green', facecolor='none', linestyle='--', label='viscosity'))
    axis.add_patch(patches.Circle((0, 0), radius=k_f,  edgecolor='yellow', facecolor='none', linestyle='--', label='forcing'))
    
    if beta != 0.0:
        axis.add_patch(patches.Circle((0, 0), radius=k_r,  edgecolor='red', facecolor='none', linestyle='--', label='drag'))
        axis.plot(+k_bx, k_by, color='cyan', linestyle='--', linewidth=1, label='beta')
        axis.plot(-k_bx, k_by, color='cyan', linestyle='--', linewidth=1)
    
    axis.legend()

    return mesh

def plot_energy_spectrum(ds, axis, t_idx, color=BLUE):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    beta = ds.attrs['model.beta']
    epsilon = A_f**2/k_f**2

    k_nu = 1/3 * n_grid
    k_b = (beta**3/epsilon)**(1/5)
    k_r = (n_grid * r_ek*beta**2/epsilon)**(1/4)

    k = ds.k.values
    Eh = ds.Eh.isel(time=t_idx)
    EEh = np.abs(ds.Eh).max().values
    loglog = axis.loglog(k, Eh/EEh, color=color)

    axis.axvline(k_nu, color='green', linestyle='--', label='viscosity')
    axis.axvline(k_f, color='yellow', linestyle='--', label='forcing')
    if beta != 0.0:
        axis.axvline(k_r, color='red', linestyle='--', label='drag')
        axis.axvline(k_b, color='cyan', linestyle='--', label='beta')

    axis.set_title(f'Energy Spectrum $\\hat E(k)$ | $ \\hat E = {EEh:.1e}$')
    axis.set_xlabel('k')
    axis.set_ylabel('E(k)')
    axis.set_ylim(1e-6, 1e1)
    axis.grid(True)
    axis.legend()
    
    return loglog

def plot_energy_x_spectrum(ds, axis, t_idx, color=BLUE):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    beta = ds.attrs['model.beta']
    epsilon = A_f**2/k_f**2

    k_nu = 1/3 * n_grid
    k_b = (beta**3/epsilon)**(1/5)
    k_r = (n_grid * r_ek*beta**2/epsilon)**(1/4)

    k = ds.kx.values
    Eh = ds.Ehx.isel(time=t_idx)
    EEh = np.abs(ds.Ehx).max().values
    loglog = axis.loglog(k, Eh/EEh, color=color)

    axis.axvline(k_nu, color='green', linestyle='--', label='viscosity')
    axis.axvline(k_f, color='yellow', linestyle='--', label='forcing')
    if beta != 0.0:
        axis.axvline(k_r, color='red', linestyle='--', label='drag')
        axis.axvline(k_b, color='cyan', linestyle='--', label='beta')

    axis.set_title(f'Energy Spectrum Along X axis $\\hat E_x(k_x)$ | $ \\hat E_x = {EEh:.1e}$')
    axis.set_xlabel('$k_x$')
    axis.set_ylabel('$E_x(k_x)$')
    axis.set_ylim(1e-6, 1e1)
    axis.grid(True)
    axis.legend()
    
    return loglog

def plot_energy_y_spectrum(ds, axis, t_idx, color=BLUE):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    beta = ds.attrs['model.beta']
    epsilon = A_f**2/k_f**2

    k_nu = 1/3 * n_grid
    k_b = (beta**3/epsilon)**(1/5)
    k_r = (n_grid * r_ek*beta**2/epsilon)**(1/4)

    k = ds.ky.values
    Eh = ds.Ehy.isel(time=t_idx)
    EEh = np.abs(ds.Ehy).max().values
    loglog = axis.loglog(k, Eh/EEh, color=color)

    k_z = np.abs(Eh.ky[Eh.argmax(dim='ky')])

    axis.axvline(k_nu, color='green', linestyle='--', label='viscosity')
    axis.axvline(k_f, color='yellow', linestyle='--', label='forcing')
    if beta != 0.0:
        axis.axvline(k_b, color='cyan', linestyle='--', label='beta')
        axis.axvline(k_r, color='red', linestyle='--', label=f'drag: k_r = {k_r:.1f}')
        axis.axvline(k_z, color='black', linestyle='--', label=f'zone: k_z = {k_z:.1f}')

    axis.set_title(f'Energy Spectrum Along Y axis $\\hat E_y(k_y)$ | $ \\hat E_y = {EEh:.1e}$')
    axis.set_xlabel('$k_y$')
    axis.set_ylabel('$E_y(k_y)$')
    axis.set_ylim(1e-6, 1e1)
    axis.grid(True)
    axis.legend()
    
    return loglog

def plot_mean_energy(ds, axis, t_idx, color=BLUE):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    epsilon = A_f**2/k_f**2 
    EEm = epsilon/(2*r_ek) / n_grid

    if t_idx != -1:
        t = ds.time.isel(time=range(t_idx)).values
        Em = ds.e_mean.isel(time=range(t_idx))
    else:
        t = ds.time.values
        Em = ds.e_mean

    loglog = axis.loglog(t, Em/EEm, color=color)
    axis.axhline(1.0, color='red', linestyle='--', label='relax')
    axis.set_title(f'Mean Energy $\\bar E(t)$ | $ \\bar E = {EEm:.1e}$')
    axis.set_xlabel('t')
    axis.set_ylabel('$\\bar E(t)$')
    axis.set_ylim(1e-3, 1e1)
    axis.grid(True)
    axis.legend()

    return loglog

def plot_x_vorticity(ds, axis, t_idx, color=GREEN):
    x = ds.x.values
    qx = ds.qx.isel(time=t_idx)
    Qx = np.abs(ds.qx).max().values
    plot = axis.plot(x, qx/Qx, color=color)
    axis.set_title(f'Vorticity Along X axis $q_x$ | $Q_x = {Qx:.1e}$')
    axis.set_xlabel('$x$')
    axis.set_ylabel('$q_x$')
    axis.set_ylim(-1, 1)
    return plot

def plot_y_vorticity(ds, axis, t_idx, color=GREEN):
    y = ds.y.values
    qy = ds.qy.isel(time=t_idx)
    Qy = np.abs(ds.qy).max().values
    plot = axis.plot(qy/Qy, y, color=color)
    axis.set_title(f'Vorticity Along Y axis $q_y$ | $Q_y = {Qy:.1e}$')
    axis.set_ylabel('$y$')
    axis.set_xlabel('$q_y$')
    axis.set_xlim(-1, 1)
    return plot

def plot_x_velocity(ds, axis, t_idx, color=BLUE):
    x = ds.x.values
    vx = ds.vx.isel(time=t_idx)
    Vx = np.abs(ds.vx).max().values
    plot = axis.plot(x, vx/Vx, color=color)
    axis.set_title(f'Velocity Along X axis $v_x$ | $V_x = {Vx:.1e}$')
    axis.set_xlabel('$x$')
    axis.set_ylabel('$v_x$')
    axis.set_ylim(-1, 1)
    return plot

def plot_y_velocity(ds, axis, t_idx, color=BLUE):
    y = ds.y.values
    vy = ds.vy.isel(time=t_idx)
    Vy = np.abs(ds.vy).max().values
    plot = axis.plot(vy/Vy, y, color=color)
    axis.set_title(f'Velocity Along Y axis $v_y$ | $V_y = {Vy:.1e}$')
    axis.set_ylabel('$y$')
    axis.set_xlabel('$v_y$')
    axis.set_xlim(-1, 1)
    return plot



PLOTS = {
    'q': plot_potential_vorticity,
    'v': plot_absolute_velocity,
    'p': plot_stream_function,
    'eh': plot_energy_field,
    'Eh': plot_energy_spectrum,
    'Ehx': plot_energy_x_spectrum,
    'Ehy': plot_energy_y_spectrum,
    'Em': plot_mean_energy,
    'qx': plot_x_vorticity,
    'qy': plot_y_vorticity,
    'vx': plot_x_velocity,
    'vy': plot_y_velocity,
}

PLOTS_COLORBAR = {
    'q': True,
    'v': True,
    'p': True,
    'eh': True,
    'Eh': False,
    'Ehx': False,
    'Ehy': False,
    'Em': False,
    'qx': False,
    'qy': False,
    'vx': False,
    'vy': False,
}