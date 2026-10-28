import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import LogNorm, Normalize
from scipy.signal import find_peaks
from .colors import *


############
# Statistics
############

def plot_relative_vorticity(ds, axis, t_idx, cmap='cmo.curl', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        q = ds.q.mean(dim='time')
        Q = np.abs(q).max().values
    else:
        q = ds.q.isel(time=t_idx)
        Q = np.abs(ds.q).max().values
    
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, q/Q, cmap=cmap, levels=scale)
    axis.set_title(f'Vorticity $q$ | $Q = {Q:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_potential_vorticity(ds, axis, t_idx, cmap='cmo.curl', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        qp = ds.qp.mean(dim='time')
        Qp = np.abs(qp).max().values
    else:
        qp = ds.qp.isel(time=t_idx)
        Qp = np.abs(ds.qp).max().values
        
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, qp/Qp, cmap=cmap, levels=scale)
    axis.set_title(f'Potential Vorticity $q_p$ | $Q_p = {Qp:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_absolute_velocity(ds, axis, t_idx, cmap='cmo.speed', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        v = ds.v.mean(dim='time')
        V = np.abs(v).max().values
    else:
        v = ds.v.isel(time=t_idx)
        V = np.abs(ds.v).max().values
        
    scale = np.linspace(0.0, +1, levels)
    cf = axis.contourf(x, y, v/V, cmap=cmap, levels=scale)
    axis.set_title(f'Absolute Velocity $v$, | $V = {V:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_zonal_velocity(ds, axis, t_idx, x0 = None, y0 = None, cmap='cmo.balance', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        ux = ds.ux.mean(dim='time')
        V = np.abs(ds.v.mean(dim='time')).max().values
    else:
        ux = ds.ux.isel(time=t_idx)
        V = np.abs(ds.v).max().values

    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, ux/V, cmap=cmap, levels=scale)
    if x0 is not None:
        axis.axvline(x[x0], color='grey', linestyle='--')
    if y0 is not None:
        axis.axhline(y[y0], color='grey', linestyle='--')

    axis.set_title(f'Zonal Velocity $u_x$, | $V = {V:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_meridional_velocity(ds, axis, t_idx, cmap='cmo.balance', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        uy = ds.uy.mean(dim='time')
        V = np.abs(ds.v.mean(dim='time')).max().values
    else:
        uy = ds.uy.isel(time=t_idx)
        V = np.abs(ds.v).max().values

    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, uy/V, cmap=cmap, levels=scale)
    axis.set_title(f'Meridional Velocity $u_y$, | $V = {V:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_stream_function(ds, axis, t_idx, cmap='cmo.delta', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        p = ds.p.mean(dim='time')
        P = np.abs(p).max().values
    else:
        p = ds.p.isel(time=t_idx)
        P = np.abs(ds.p).max().values
    
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, p/P, cmap=cmap, levels=scale)
    axis.set_title(f'Stream Function $\\psi$ | $\\Psi = {P:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_energy_field(ds, axis, t_idx, cmap='cmo.dense', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        e = ds.e.mean(dim='time')
        E = np.abs(e).max().values
    else:
        e = ds.e.isel(time=t_idx)
        E = np.abs(ds.e).max().values

    scale = np.linspace(-4, 0, levels)
    cf = axis.contourf(x, y, np.log10(e/E), cmap=cmap, levels=scale)
    axis.set_title(f'Energy Field $\\log e$ | $E = {E:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_enstrophy_field(ds, axis, t_idx, cmap='cmo.deep', levels=50):
    x = ds.x.values
    y = ds.y.values
    if t_idx is None:
        z = ds.z.mean(dim='time')
        Z = np.abs(z).max().values
    else:
        z = ds.z.isel(time=t_idx)
        Z = np.abs(ds.z).max().values

    scale = np.linspace(-4, 0, levels)
    cf = axis.contourf(x, y, np.log10(z/Z), cmap=cmap, levels=scale)
    axis.set_title(f'Enstrophy Field $\\log z$ | $Z = {Z:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_x_velocity(ds, axis, t_idx, y0 = 0, color=BLUE):
    n_grid = len(ds.x.values)
    x = ds.x.values
    y = ds.y.values
    uy = ds.uy.isel(time=t_idx)
    uy = uy.isel(y=y0)
    V = np.abs(ds.v).max().values
    vy = uy/V

    plot = axis.plot(x, vy, color='black', lw=0.5)
    axis._fill_between_x_or_y("x", x, 0, vy, where=(vy > 0), color='red', alpha=0.5)
    axis._fill_between_x_or_y("x", x, 0, vy, where=(vy < 0), color='blue', alpha=0.5)
    axis.axvline(0.0, color='black', linestyle='--', label='zero')

    axis.set_title(f'Velocity Along Y axis $v_y$ | $V = {V:.1e}$')
    axis.set_xlabel('$v_y$')
    axis.set_ylabel('$y$')
    axis.set_xlim(0, 2*np.pi)
    axis.set_ylim(-1, 1)
    return plot


def plot_y_velocity(ds, axis, t_idx, x0 = 0, color=BLUE):
    n_grid = len(ds.x.values)
    y = ds.y.values
    ux = ds.ux.isel(time=t_idx)
    ux = ux.isel(x=x0)
    V = np.abs(ds.v).max().values
    vx = ux/V

    plot = axis.plot(vx, y, color='black', lw=0.5)
    axis.fill_betweenx(y, 0, vx, where=(vx > 0), color='red', alpha=0.5)
    axis.fill_betweenx(y, 0, vx, where=(vx < 0), color='blue', alpha=0.5)
    axis.axvline(0.0, color='black', linestyle='--', label='zero')

    axis.set_title(f'Velocity Along X axis $v_x$ | $V = {V:.1e}$')
    axis.set_xlabel('$x$')
    axis.set_ylabel('$v_x$')
    axis.set_xlim(-1, 1)
    axis.set_ylim(0, 2*np.pi)
    return plot

def plot_energy_spectrum(ds, axis, t_idx, cmap='cmo.dense', levels=100):
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
    if t_idx is None:
        eh = ds.eh.mean(dim='time')
        Eh = np.abs(eh).max().values
    else:
        eh = ds.eh.isel(time=t_idx)
        Eh = np.abs(ds.eh).max().values
        
    norm = Normalize(vmin=-6.0, vmax=1.0)
    mesh = axis.pcolormesh(kx, ky, np.log10(eh/Eh), cmap=cmap, norm=norm)
    axis.set_title(f'Energy Spectrum $\\hat e$ | $ \\hat E = {Eh:.1e}$')
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

def plot_radial_energy_spectrum(ds, axis, t_idx, color=BLUE):
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
    if t_idx is None:
        Ehr = ds.Ehr.mean(dim='time')
        EEhr = np.abs(Ehr).max().values
    else:
        Ehr = ds.Ehr.isel(time=t_idx)
        EEhr = np.abs(ds.Ehr).max().values

    loglog = axis.loglog(k, Ehr/EEhr, color=color)
    axis.set_title(f'Radial Energy Spectrum $\\hat e_r$ | $ \\hat E_r = {EEhr:.1e}$')
    axis.set_xlabel('k')
    axis.set_ylabel('E(k)')

    axis.axvline(k_nu, color='green', linestyle='--', label='viscosity')
    axis.axvline(k_f, color='yellow', linestyle='--', label='forcing')
    if beta != 0.0:
        axis.axvline(k_r, color='red', linestyle='--', label='drag')
        axis.axvline(k_b, color='cyan', linestyle='--', label='beta')
    axis.set_ylim(1e-6, 1e1)
    axis.grid(True)
    axis.legend()
    return loglog

def plot_polarized_energy_spectrum(ds, axis, t_idx, color=BLUE):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    beta = ds.attrs['model.beta']
    epsilon = A_f**2/k_f**2

    k_nu = 1/3 * n_grid
    k_b = (beta**3/epsilon)**(1/5)
    k_r = (n_grid * r_ek*beta**2/epsilon)**(1/4)

    kx = ds.kx.values
    ky = ds.ky.values
    if t_idx is None:
        Ehx = ds.Ehx.mean(dim='time')
        Ehy = ds.Ehy.mean(dim='time')
        EEhx = np.abs(Ehx).max().values
        EEhy = np.abs(Ehy).max().values
    else:
        Ehx = ds.Ehx.isel(time=t_idx)
        Ehy = ds.Ehy.isel(time=t_idx)
        EEhx = np.abs(ds.Ehx).max().values
        EEhy = np.abs(ds.Ehy).max().values

    EEh = max(EEhx, EEhy)
    loglog = axis.loglog(kx, Ehx/EEh, color=RED, label='zonal (kx)')
    loglog = axis.loglog(ky, Ehy/EEh, color=BLUE, label='meridional (ky)')

    axis.set_title(f'Polarized Fourier Spectrum $\\hat e_p$ | $ \\hat E_p = {EEhy:.1e}$')
    axis.set_xlabel('k')
    axis.set_ylabel('E(k)')

    axis.axvline(k_nu, color='green', linestyle='--', label='viscosity')
    axis.axvline(k_f, color='yellow', linestyle='--', label='forcing')
    if beta != 0.0:
        axis.axvline(k_r, color='red', linestyle='--', label='drag')
        axis.axvline(k_b, color='cyan', linestyle='--', label='beta')
    axis.set_ylim(1e-6, 1e1)
    axis.grid(True)
    axis.legend()
    return loglog

def plot_energy_zonal_scalogram(ds, axis, t_idx, cmap='cmo.dense', levels=100):
    kx = ds.kx.values
    y = ds.y.values
    if t_idx is None:
        ehx = ds.ehx.mean(dim='time')
        Ehx = np.abs(ehx).max().values
    else:
        ehx = ds.ehx.isel(time=t_idx)
        Ehx = np.abs(ds.ehx).max().values
        
    norm = Normalize(vmin=-6.0, vmax=1.0)
    mesh = axis.pcolormesh(kx, y, np.log10(ehx/Ehx), cmap=cmap, norm=norm)
    axis.set_title(f'Zonal Fourier Spectrum $\\hat e_z$ | $ \\hat E_z = {Ehx:.1e}$')
    axis.set_xlabel('Kx')
    axis.set_ylabel('Y')
    axis.set_xscale('log')
    return mesh

def plot_energy_meridional_scalogram(ds, axis, t_idx, cmap='cmo.dense', levels=100):
    x = ds.x.values
    ky = ds.ky.values
    if t_idx is None:
        ehy = ds.ehy.mean(dim='time')
        Ehy = np.abs(ehy).max().values
    else:
        ehy = ds.ehy.isel(time=t_idx)
        Ehy = np.abs(ds.ehy).max().values
        
    norm = Normalize(vmin=-6.0, vmax=1.0)
    mesh = axis.pcolormesh(x, ky, np.log10(ehy/Ehy), cmap=cmap, norm=norm)
    axis.set_title(f'Meridional Fourier Spectrum $\\hat e_m$ | $ \\hat E_m = {Ehy:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Ky')
    axis.set_yscale('log')
    return mesh

def plot_energy_swt_field(ds, axis, t_idx, s_idx=None, cmap='cmo.dense', levels=50, swt=None):
    if s_idx is None:
        if swt == 0:
            ds = ds.sum(dim='s')
            J = 'A'
        else:
            ds = ds.isel(s=-swt)
            J = round(np.log(ds.s.values)/np.log(2))
    else:
        ds = ds.isel(s=s_idx)
        J = round(np.log(ds.s.values)/np.log(2))

    x = ds.x.values
    y = ds.y.values

    if t_idx is None:
        ew = ds.ew.mean(dim='time')
        Ew = np.abs(ew).max().values
    else:
        ew = ds.ew.isel(time=t_idx)
        Ew = np.abs(ds.ew).max().values

    scale = np.linspace(-4, 0, levels)
    cf = axis.contourf(x, y, np.log10(ew/Ew), cmap=cmap, levels=scale)
    axis.set_title(f'Energy Field [J = {J}] $\\log e_{J}$ | $E_{J} = {Ew:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_energy_swt_zonal_scalogram(ds, axis, t_idx, s_idx=None, cmap='cmo.dense', levels=100):
    if s_idx is None:
        ds = ds.mean(dim='x')
    else:
        ds = ds.isel(x=s_idx)

    # kx = ds.attrs['grid.n_grid']/ds.s.values
    kx = 2 * np.pi/ds.s.values
    y = ds.y.values
    if t_idx is None:
        ewh = ds.ewh.mean(dim='time')
        Ewh = np.abs(ewh).max().values
    else:
        ewh = ds.ewh.isel(time=t_idx)
        Ewh = np.abs(ds.ewh).max().values
        
    norm = Normalize(vmin=-6.0, vmax=1.0)
    mesh = axis.pcolormesh(kx, y, np.log10(ewh/Ewh), cmap=cmap, norm=norm, shading='nearest')
    axis.set_title(f'Zonal Wavelet Spectrum $\\hat e_z$ | $ \\hat E_z = {Ewh:.1e}$')
    axis.set_xlabel('Kx')
    axis.set_ylabel('Y')
    axis.set_xscale('log')
    return mesh

def plot_energy_swt_meridional_scalogram(ds, axis, t_idx, s_idx=None, cmap='cmo.dense', levels=100):
    if s_idx is None:
        ds = ds.mean(dim='y')
    else:
        ds = ds.isel(y=s_idx)

    x = ds.x.values
    # ky = ds.attrs['grid.n_grid']/ds.s.values
    ky = 2 * np.pi/ds.s.values
    if t_idx is None:
        ewv = ds.ewv.mean(dim='time')
        Ewy = np.abs(ewv).max().values
    else:
        ewv = ds.ewv.isel(time=t_idx)
        Ewy = np.abs(ds.ewv).max().values
        
    norm = Normalize(vmin=-6.0, vmax=1.0)
    mesh = axis.pcolormesh(x, ky, np.log10(ewv/Ewy).T, cmap=cmap, norm=norm)
    axis.set_title(f'Meridional Wavelet Spectrum $\\hat e_m$ | $ \\hat E_m = {Ewy:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Ky')
    axis.set_yscale('log')
    return mesh

def plot_polarized_swt_energy_spectrum(ds, axis, t_idx, color=BLUE):
    n_grid = ds.attrs['grid.n_grid']
    A_f = ds.attrs['model.af']
    k_f = ds.attrs['model.kf']
    r_ek = ds.attrs['model.rek']
    beta = ds.attrs['model.beta']
    epsilon = A_f**2/k_f**2

    k_nu = 1/3 * n_grid
    k_b = (beta**3/epsilon)**(1/5)
    k_r = (n_grid * r_ek*beta**2/epsilon)**(1/4)

    # kx = ds.kx.values
    # ky = ds.ky.values
    kx = 2 * np.pi/ds.s.values
    ky = 2 * np.pi/ds.s.values

    if t_idx is None:
        Ewh = ds.Ewh.mean(dim='time')
        Ewv = ds.Ewv.mean(dim='time')
        EEwh = np.abs(Ewh).max().values
        EEwv = np.abs(Ewv).max().values
    else:
        Ewh = ds.Ewh.isel(time=t_idx)
        Ewv = ds.Ewv.isel(time=t_idx)
        EEwh = np.abs(ds.Ewh).max().values
        EEwv = np.abs(ds.Ewv).max().values

    EEw = max(EEwh, EEwv)

    loglog = axis.loglog(kx, Ewh/EEw, color=RED, label='zonal (kx)')
    loglog = axis.loglog(ky, Ewv/EEw, color=BLUE, label='meridional (ky)')

    axis.set_title(f'Polarized Wavelet Spectrum $\\hat e_p$ | $ \\hat E_p = {EEwh:.1e}$')
    axis.set_xlabel('k')
    axis.set_ylabel('E(k)')

    axis.axvline(k_nu, color='green', linestyle='--', label='viscosity')
    axis.axvline(k_f, color='yellow', linestyle='--', label='forcing')
    if beta != 0.0:
        axis.axvline(k_r, color='red', linestyle='--', label='drag')
        axis.axvline(k_b, color='cyan', linestyle='--', label='beta')
    axis.set_ylim(1e-6, 1e1)
    axis.grid(True)
    axis.legend()
    return loglog

PLOTS = {
    'q': plot_relative_vorticity,
    'qp': plot_potential_vorticity,
    'v': plot_absolute_velocity,
    'ux': plot_zonal_velocity,
    'uy': plot_meridional_velocity,
    'vx': plot_x_velocity,
    'vy': plot_y_velocity,
    'p': plot_stream_function, 
    'e': plot_energy_field,
    'eh': plot_energy_spectrum,
    'ehx': plot_energy_zonal_scalogram,
    'ehy': plot_energy_meridional_scalogram,
    'Ehr': plot_radial_energy_spectrum,
    'Ehp': plot_polarized_energy_spectrum,
    'ew': plot_energy_swt_field,
    'ewz': plot_energy_swt_zonal_scalogram,
    'ewm': plot_energy_swt_meridional_scalogram,
    'Ewp': plot_polarized_swt_energy_spectrum,
}

PLOTS_COLORBAR = {
    'q': True,
    'qp': True,
    'v': True,
    'ux': True,
    'uy': True,
    'vx': False,
    'vy': False,
    'p': True,
    'e': True,
    'eh': True,
    'ehx': True,
    'ehy': True,
    'Ehr': False,
    'Ehp': False,
    'ew': True,
    'ewz': True,
    'ewm': True,
    'Ewp': False,
}

FIELDS = {
    'q': 'q',
    'Q': 'qp',
    'U': 'v',
    'u': 'ux',
    'v': 'uy',
    # 'p': 'p',
    'x': 'vx',
    'y': 'vy',
    'e': 'e',
    'E': 'eh',
    'Z': 'ehx',
    'M': 'ehy',
    'R': 'Ehr',
    'P': 'Ehp',
    'w': 'ew',
    'z': 'ewz',
    'm': 'ewm',
    'p': 'Ewp',
}