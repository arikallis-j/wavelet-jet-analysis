import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cmocean.cm as cmo

from matplotlib.colors import LogNorm, Normalize

from scipy.signal import find_peaks



YELLOW = '#ffff99'
GREEN = '#7fc87f'
BLUE = '#385eb1'
RED = '#e04c4c'

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

def plot_zonal_velocity(ds, axis, t_idx, cmap='cmo.balance', levels=50):
    # x0_idx = 128
    x = ds.x.values
    y = ds.y.values
    Ux = np.abs(ds.ux).max().values
    ux = ds.ux.isel(time=t_idx)
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, ux/Ux, cmap=cmap, levels=scale)
    y_jet = ds.y_jet.isel(time=t_idx).values
    y_ajet = ds.y_ajet.isel(time=t_idx).values
    y_njet = ds.y_njet.isel(time=t_idx).values
    y_jet_m = ds.y_jet.isel(time=t_idx).mean(dim='x').values
    y_ajet_m = ds.y_ajet.isel(time=t_idx).mean(dim='x').values
    n_jets = y_jet.shape[0]
    
    for k in range(n_jets):
        axis.scatter(x, y_jet[k, :], c='red', s=1, marker='s')
        axis.scatter(x, y_ajet[k, :], c='blue', s=1, marker='s')
        axis.scatter(x, y_njet[k, :], c='grey', s=1, marker='s')
        axis.axhline(y_jet_m[k], color='red', linestyle='--')
        axis.axhline(y_ajet_m[k], color='blue', linestyle='--')

    axis.set_title(f'Zonal Velocity $u_x$, | $U_x = {Ux:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

# def plot_zonal_velocity(ds, axis, t_idx, cmap='cmo.balance', levels=50):
#     alpha = 0.0
#     x = ds.x.values
#     y = ds.y.values
#     Ux = np.abs(ds.ux).max().values
#     ux = ds.ux.isel(time=t_idx)
#     scale = np.linspace(-1.0, +1, levels)
#     cf = axis.contourf(x, y, ux/Ux, cmap=cmap, levels=scale)

#     y_jets = trace_jet_ridges(ux.values/Ux, n_jets=3, threshold_percentile=70, smooth_sigma=1.0)
#     for y_j in y_jets:
#         axis.plot(x, y_j, color='yellow')
    
#     axis.set_title(f'Zonal Velocity $u_x$, | $U_x = {Ux:.1e}$')
#     axis.set_xlabel('X')
#     axis.set_ylabel('Y')
#     return cf

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
    if t_idx == -1:
        Eh = ds.Ehx.mean(dim='time')
    else:
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
    if t_idx == -1:
        Eh = ds.Ehy.mean(dim='time')
    else:
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

def plot_energy_xy_spectrum(ds, axis, t_idx, color=BLUE):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    beta = ds.attrs['model.beta']
    epsilon = A_f**2/k_f**2

    k_nu = 1/3 * n_grid
    k_b = (beta**3/epsilon)**(1/5)
    k_r = 1/2 * (n_grid * r_ek*beta**2/epsilon)**(1/4)

    EEh = np.abs(ds.Ehy.mean('time')).max().values
    # EEw = np.abs(ds.Ewh.mean('time')).max().values

    # k = ds.k.values
    # if t_idx == -1:
    #     Eh = ds.Eh.mean(dim='time')
    # else:
    #     Eh = ds.Eh.isel(time=t_idx)
    # loglog = axis.loglog(k, Eh/EEh, color='black', label='fA')

    # k_freq = n_grid / ds.s.values 

    # if t_idx == -1:
    #     Ew = ds.Ewv.mean(dim='time')
    # else:
    #     Ew = ds.Ewv.isel(time=t_idx)
    # loglog = axis.loglog(k_freq, Ew/EEw, color=BLUE, label='dwt meridional', linestyle='--')


    # if t_idx == -1:
    #     Ew = ds.Ewh.mean(dim='time')
    # else:
    #     Ew = ds.Ewh.isel(time=t_idx)
    # loglog = axis.loglog(k_freq, Ew/EEw, color=RED, label='dwt zonal', linestyle='--')



    k = ds.ky.values
    if t_idx == -1:
        Eh = ds.Ehy.mean(dim='time')
    else:
        Eh = ds.Ehy.isel(time=t_idx)
    loglog = axis.loglog(k, Eh/EEh, color=BLUE, label='meridional')
    k_z = np.abs(ds.ky[Eh.argmax(dim='ky')])

    k = ds.kx.values
    if t_idx == -1:
        Eh = ds.Ehx.mean(dim='time')
    else:
        Eh = ds.Ehx.isel(time=t_idx)
    loglog = axis.loglog(k, Eh/EEh, color=RED, label='zonal')


    axis.axvline(k_nu, color='green', linestyle='--', label='viscosity')
    axis.axvline(k_f, color='yellow', linestyle='--', label='forcing')
    if beta != 0.0:
        axis.axvline(k_b, color='cyan', linestyle='--', label='beta')
        axis.axvline(k_r, color='red', linestyle='--', label=f'drag: k_r = {k_r:.1f}')
        axis.axvline(k_z, color='grey', linestyle='--', label=f'jet: {k_z:.0f}')
        # axis.axvline(2*k_z, color='grey', linestyle='--', label=f'jet: {2*k_z:.0f}')

    axis.set_title(f'Energy Spectrum $\\hat E(k)$ | $ \\hat E = {EEh:.1e}$')
    axis.set_xlabel('$k$')
    axis.set_ylabel('$E(k)$')
    axis.set_ylim(1e-6, 1e1)
    axis.grid(True)
    axis.legend()
    
    return loglog

# def plot_energy_y_spectrum(ds, axis, t_idx, color=BLUE):
#     n_grid = ds.attrs['grid.n_grid']
#     A_f = ds.attrs['model.af']
#     k_f = ds.attrs['model.kf']
#     r_ek = ds.attrs['model.rek']
#     beta = ds.attrs['model.beta']
#     epsilon = A_f**2/k_f**2

#     k_nu = 1/3 * n_grid
#     k_b = (beta**3/epsilon)**(1/5)
#     k_r = (n_grid * r_ek*beta**2/epsilon)**(1/4)

#     # k = ds.ky.values
#     if t_idx == -1:
#         Eh = ds.Eh.mean(dim='time')
#         Ehh = ds.Ehx.mean(dim='time')
#         Ehv = ds.Ehy.mean(dim='time')
#     else:
#         Eh = ds.Ehy.isel(time=t_idx)

#     k = Eh.k.values
#     kx = Ehh.kx.values
#     ky = Ehv.ky.values
#     EEh = np.abs(ds.Eh).max().values

#     loglog = axis.loglog(k, Eh/EEh, color='black', label='Fourier (A)')
#     loglog = axis.loglog(kx, Ehh/EEh, color=RED, label='Fourier (H)')
#     loglog = axis.loglog(ky, Ehv/EEh, color=BLUE, label='Fourier (V)')

#     axis.set_title(f'Energy Spectrum $\\hat E(k)$ | $ \\hat E = {EEh:.1e}$')
#     axis.set_xlabel('$k$')
#     axis.set_ylabel('$E(k)$')
#     axis.set_ylim(1e-6, 1e1)
#     axis.grid(True)
#     axis.legend()
    
#     return loglog

def plot_jet_spectrum(ds, axis, t_idx, color=BLUE):
    x0_idx = 128
    n_grid = ds.attrs['grid.n_grid']
    n_jets = len(ds.k_jet.values)

    if t_idx == -1:
        eh_jet = ds.eh_jet.mean(dim=['time'])
        eh_ajet = ds.eh_ajet.mean(dim=['time'])
        ew_jet = ds.ew_jet.mean(dim=['x', 'time'])
        ew_ajet = ds.ew_ajet.mean(dim=['x', 'time'])
        # ew_njet = ds.ew_njet.mean(dim=['x', 'time'])
        ewh_jet = ds.ewh_jet.mean(dim=['x', 'time'])
        ewh_ajet = ds.ewh_ajet.mean(dim=['x', 'time'])
        # ewh_njet = ds.ewh_njet.mean(dim=['x', 'time'])
        # ewv_jet = ds.ewv_jet.mean(dim=['x', 'time'])
        # ewv_ajet = ds.ewv_ajet.mean(dim=['x', 'time'])
        # ewv_njet = ds.ewv_njet.mean(dim=['x', 'time'])
        # ewd_jet = ds.ewd_jet.mean(dim=['x', 'time'])
        # ewd_ajet = ds.ewd_ajet.mean(dim=['x', 'time'])
        # ewd_njet = ds.ewd_njet.mean(dim=['x', 'time'])
    else:
        eh_jet = ds.eh_jet.isel(time=t_idx)
        eh_ajet = ds.eh_ajet.isel(time=t_idx)
        ew_jet = ds.ew_jet.isel(time=t_idx).mean(dim='x')
        ew_ajet = ds.ew_ajet.isel(time=t_idx).mean(dim='x')
    #     ew_njet = ds.ew_njet.isel(time=t_idx).mean(dim='x')
        ewh_jet = ds.ewh_jet.isel(time=t_idx).mean(dim='x')
        ewh_ajet = ds.ewh_ajet.isel(time=t_idx).mean(dim='x')
    #     ewh_njet = ds.ewh_njet.isel(time=t_idx).mean(dim='x')
    #     ewv_jet = ds.ewv_jet.isel(time=t_idx).mean(dim='x')
    #     ewv_ajet = ds.ewv_ajet.isel(time=t_idx).mean(dim='x')
    #     ewv_njet = ds.ewv_njet.isel(time=t_idx).mean(dim='x')
    #     ewd_jet = ds.ewd_jet.isel(time=t_idx).mean(dim='x')
    #     ewd_ajet = ds.ewd_ajet.isel(time=t_idx).mean(dim='x')
    #     ewd_njet = ds.ewd_njet.isel(time=t_idx).mean(dim='x')

    k_freq = n_grid / ds.s.values 
    EEw = np.abs(ds.ew_jet.mean(dim='x')).max().values
    kx = ds.kx.values 
    EEh = np.abs(ds.eh_jet).max().values
    loglog = axis.loglog(kx[kx>0], eh_jet[kx>0]/EEh, label='Jet (FH)', color=BLUE, linestyle=':')
    loglog = axis.loglog(kx[kx>0], eh_ajet[kx>0]/EEh, label='Anti-jet (FH)', color=RED, linestyle=':')

    loglog = axis.loglog(k_freq, ew_jet/EEw, label='Jet (A)', color=BLUE, linestyle='-')
    loglog = axis.loglog(k_freq, ewh_jet/EEw, label='Jet (H)', color=BLUE, linestyle='--')
    # loglog = axis.loglog(k_freq, ewv_jet/EEw, label='Jet (V)', color=BLUE, linestyle='-.')
    # loglog = axis.loglog(k_freq, ewd_jet/EEw, label='Jet (D)', color=BLUE, linestyle=':')
    loglog = axis.loglog(k_freq, ew_ajet/EEw, label='Anti-jet (A)', color=RED, linestyle='-')
    loglog = axis.loglog(k_freq, ewh_ajet/EEw, label='Anti-jet (H)', color=RED, linestyle='--')
    # loglog = axis.loglog(k_freq, ewv_ajet/EEw, label='Anti-jet (V)', color=RED, linestyle='-.')
    # loglog = axis.loglog(k_freq, ewd_ajet/EEw, label='Anti-jet (D)', color=RED, linestyle=':')
    # loglog = axis.loglog(k_freq, ew_njet/EEw, label='No-jet (A)', color='grey', linestyle='-')
    # loglog = axis.loglog(k_freq, ewh_njet/EEw, label='No-jet (H)', color='grey', linestyle='--')
    # loglog = axis.loglog(k_freq, ewv_njet/EEw, label='No-jet (V)', color='grey', linestyle='-.')
    # loglog = axis.loglog(k_freq, ewd_njet/EEw, label='No-jet (D)', color='grey', linestyle=':')

    # axis.axvline(k_z, color='black', linestyle='--', label=f'k-jets = {k_z:.0f}')

    axis.set_title(f'Energy Jet Spectrum $E_j(k)$ | $ E_j = {EEh:.1e}$')
    axis.set_xlabel('$k$')
    axis.set_ylabel('$E_j(k)$')
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
    y = ds.y.values
    vx = ds.vx.isel(time=t_idx)
    V = np.abs(ds.v).max().values
    v = vx/V
    plot = axis.plot(v, y, color='black', lw=0.5)
    axis.fill_betweenx(y, 0, v, where=(v > 0), color='red', alpha=0.5)
    axis.fill_betweenx(y, 0, v, where=(v < 0), color='blue', alpha=0.5)
    axis.axvline(0.0, color='black', linestyle='--', label='zero')
    axis.set_title(f'Velocity Along X axis $v_x$ | $V = {V:.1e}$')
    axis.set_ylabel('$y$')
    axis.set_xlabel('$v_x$')
    axis.axvline(0.0, color='black', linestyle='--', label='zero')
    axis.set_xlim(-1, 1)
    axis.set_ylim(0, 2*np.pi)
    return plot

# def plot_y_velocity(ds, axis, t_idx, color=BLUE):
#     x = ds.x.values
#     vy = ds.vy.isel(time=t_idx)
#     V = np.abs(ds.v).max().values
#     plot = axis.plot(x, vy/V, color=color)
#     axis.set_title(f'Velocity Along Y axis $v_y$ | $V = {V:.1e}$')
#     axis.set_xlabel('$x$')
#     axis.set_ylabel('$v_y$')
#     axis.set_ylim(-1, 1)
#     return plot

def plot_y_velocity(ds, axis, t_idx, color=BLUE):
    n_grid = len(ds.x.values)
    n_jets = 3
    x0 = n_grid//2
    y = ds.y.values
    ux = ds.ux.isel(time=t_idx)
    ux = ux.isel(x=x0)
    V = np.abs(ds.v).max().values
    vx = ux/V

    vx_periodic = np.concatenate([vx, vx])
    peaks_idx, _ = find_peaks(vx_periodic, distance=n_grid//(2*n_jets))
    peaks_idx = peaks_idx[peaks_idx < n_grid]
    jets_idx = peaks_idx[np.argsort(vx[peaks_idx])[::-1][:n_jets]]
    # print(jets_idx)

    avx_periodic = np.concatenate([-vx, -vx])
    peaks_idx, _ = find_peaks(avx_periodic, distance=n_grid//(2*n_jets))
    peaks_idx = peaks_idx[peaks_idx < n_grid]
    ajets_idx = peaks_idx[np.argsort(-vx[peaks_idx])[::-1][:n_jets]]
    # print(ajets_idx)

    plot = axis.plot(vx, y, color='black', lw=0.5)
    axis.fill_betweenx(y, 0, vx, where=(vx > 0), color='red', alpha=0.5)
    axis.fill_betweenx(y, 0, vx, where=(vx < 0), color='blue', alpha=0.5)
    axis.axvline(0.0, color='black', linestyle='--', label='zero')
    for jet_idx in jets_idx:
        axis.axhline(y[jet_idx], color='red', linestyle='--', label='jets', alpha=0.5)
    for ajet_idx in ajets_idx:
        axis.axhline(y[ajet_idx], color='blue', linestyle='--', label='ajets', alpha=0.5)

    axis.set_title(f'Velocity Along X axis $v_x$ | $V = {V:.1e}$')
    axis.set_xlabel('$x$')
    axis.set_ylabel('$v_x$')
    axis.set_xlim(-1, 1)
    axis.set_ylim(0, 2*np.pi)
    return plot



def plot_swt_field(ds, axis, t_idx, j=1, swt=0, cmap='cmo.dense', levels=50):
    J = (j+swt)
    n_grid = ds.attrs['grid.n_grid']
    # n_lev = ds.attrs['grid.n_lev']
    x = ds.x.values
    y = ds.y.values
    ew_j = ds.ew.isel(s=J - 1)
    ew = ew_j.isel(time=t_idx)
    Ew = np.abs(ew_j).max().values
    k = n_grid/ew_j.s.values
    # norm = LogNorm(vmin=1e-6, vmax=1e1)
    # norm = Normalize(vmin=-6.0, vmax=1.0)
    # mesh = axis.pcolormesh(x, y, np.log10(ew/Ew), cmap=cmap, norm=norm)
    scale = np.linspace(-6.0, 1.0, levels)
    mesh = axis.contourf(x, y, np.log10(ew/Ew), cmap=cmap, levels=scale)
    axis.set_title(f'Energy $\log$ e{J}$(x, y)$ | E{J} = {Ew:.1e}, k{J} = {k:.0f}')
    # axis.set_aspect('equal')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return mesh

PLOTS = {
    'q': plot_potential_vorticity,
    'v': plot_absolute_velocity,
    'ux': plot_zonal_velocity,
    'p': plot_stream_function,
    'eh': plot_energy_field,
    'Eh': plot_energy_spectrum,
    'Ehx': plot_energy_x_spectrum,
    'Ehy': plot_energy_y_spectrum,
    'Ehxy': plot_energy_xy_spectrum,
    'Em': plot_mean_energy,
    'Ej': plot_jet_spectrum,
    'qx': plot_x_vorticity,
    'qy': plot_y_vorticity,
    'vx': plot_x_velocity,
    'vy': plot_y_velocity,
    'ew': plot_swt_field,
}

PLOTS_COLORBAR = {
    'q': True,
    'v': True,
    'ux': True,
    'p': True,
    'eh': True,
    'Eh': False,
    'Ehx': False,
    'Ehy': False,
    'Ehxy': False,
    'Em': False,
    'Ej': False,
    'qx': False,
    'qy': False,
    'vx': False,
    'vy': False,
    'ew': True
}

FIELDS = {
    'q': 'q',
    'v': 'v',
    'u': 'ux',
    'p': 'p',
    'e': 'eh',
    'E': 'Eh',
    'X': 'Ehx',
    'Y': 'Ehy',
    'Z': 'Ehxy',
    'm': 'Em',
    'j': 'Ej',
    'x': 'qx',
    'y': 'qy',
    'k': 'vx',
    'l': 'vy',
    'w': 'ew',
}