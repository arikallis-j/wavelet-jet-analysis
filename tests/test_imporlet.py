import typer
import qgtoolbox as qg
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xwavelet.wavelet as xwl
import xrft

from tqdm import tqdm
from matplotlib.colors import LogNorm, Normalize

def plot_morlet2_map(ds, s_idx, th_idx, gamma, path):
    x, y = ds.x, ds.y
    ntheta = 50
    a = 1.0
    n_grid = ds.attrs['grid.n_grid']
    n_scale = n_grid
    x0 = ds.x.values[1] - ds.x.values[0]
    s_vals = np.geomspace(2.0, n_grid, n_scale)
    s = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})
    s = s.isel(s=[s_idx])

    morlet2, phi = xwl._shift_morlet2(x0, ntheta, a, s, y, x, gamma)
    if th_idx is None:
        morlet2 = morlet2.mean(dim='angle')
    else:
        morlet2 = morlet2.isel(angle=th_idx)
    morlet2 = morlet2.isel(s=0) 
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    if th_idx is None:
        fig.suptitle(f"2D Morlet | s = {morlet2.s.values:0.1f}")
    else:
        fig.suptitle(f"2D Morlet | s = {morlet2.s.values:0.1f} | theta = {morlet2.angle.values / (np.pi)*180 :0.1f} | gamma = ({gamma[0]/(np.pi):0.1f}$\\pi$, {gamma[1]/(np.pi):0.1f}$\\pi$)")

    axis = axes[0]
    mesh = axis.pcolormesh(x, y, morlet2.real, vmin=-1.0, vmax=1.0)
    axis.set_title(f'Real')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    axis = axes[1]
    mesh = axis.pcolormesh(x, y, morlet2.imag, vmin=-1.0, vmax=1.0)
    axis.set_title(f'Imag')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    axis = axes[2]
    mesh = axis.pcolormesh(x, y, np.abs(morlet2), vmin=0.0, vmax=1.0)
    axis.set_title(f'Abs')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()

def plot_gamma(ds, gamma, path):
    fig, axis = plt.subplots(figsize=(7, 7))
    qg.plot_zonal_velocity(ds, axis, t_idx=0)
    axis.axvline(gamma[0], color='grey', linestyle='--', alpha=0.5)
    axis.axhline(gamma[1], color='grey', linestyle='--', alpha=0.5)
    axis.scatter(gamma[0], gamma[1], color='black', marker='+', s=1000)
    fig.savefig(path / 'ux-gamma.png', dpi=150)
    plt.show()
    plt.close()

    fig, axis = plt.subplots(figsize=(7, 7))
    qg.plot_relative_vorticity(ds, axis, t_idx=0)
    axis.axvline(gamma[0], color='grey', linestyle='--', alpha=0.5)
    axis.axhline(gamma[1], color='grey', linestyle='--', alpha=0.5)
    axis.scatter(gamma[0], gamma[1], color='black', marker='+', s=1000)
    fig.savefig(path / 'q-gamma.png', dpi=150)
    plt.show()
    plt.close()


def main(n_epoch: int = 1, exp: str = 'test', n_scale: int = 10):
    dm = qg.DataManager()
    sim_path = dm.make_dir(path=dm.raw, name=exp)
    res_path = dm.make_dir(path=dm.processed, name=exp)
    ds_sim_path = sim_path / f"{exp}.nc"
    ds = qg.load_dataset(ds_sim_path)
    ds = qg.calc_statistics(ds)
    th_idx = 0
    gamma_idx = (92, 121)
    gamma = (ds.x.isel(x=gamma_idx[0]).values, ds.y.isel(y=gamma_idx[1]).values)
    plot_gamma(ds, gamma, res_path)

    x, y = ds.x, ds.y  
    ntheta = 2 * n_scale
    a = 1.0
    n_grid = ds.attrs['grid.n_grid']
    n_scale = n_scale # n_grid
    x0 = ds.x.values[1] - ds.x.values[0]
    s_vals = np.geomspace(2.0, n_grid, n_scale)
    s = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})
    
    da = ds.q.isel(time=0)
    Wa = xwl.cwt2(da, s, dim=['y', 'x'], x0=x0, ntheta=ntheta, gamma=gamma, wtype='imorlet', center=False)
    Eai = np.abs(Wa)**2 / Wa["s"] * x0**2

    Wa = xwl.cwt2(da, s, dim=['y', 'x'], x0=x0, ntheta=ntheta, gamma=gamma, wtype='smorlet', center=False)
    Ea = np.abs(Wa)**2 / Wa["s"] * x0**2
    k = 2 * np.pi/Ea.s.values
    theta = Ea.angle.values

    Eai = Eai
    Eaz = Ea.sel(angle=0.0, method='nearest')
    Eam = Ea.sel(angle=np.pi/2, method='nearest')
    Eaa = Ea.mean(dim='angle')

    fig, axis = plt.subplots(figsize=(7, 7))
    loglog = axis.loglog(k, Eaz, color=qg.RED, label='zonal (kx)')
    loglog = axis.loglog(k, Eam, color=qg.BLUE, label='meridional (ky)')
    loglog = axis.loglog(k, Eai, color=qg.RED, label='isotropic', linestyle='--')
    loglog = axis.loglog(k, Eaa, color=qg.BLUE, label='averaged', linestyle='--')

    axis.set_title(f'Enstrophy Wavelet Spectrum $\\tilde Z$')
    axis.set_xlabel('k')
    axis.set_ylabel('Z(k)')
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(res_path / 'Z-gamma.png', dpi=150)
    plt.show()
    plt.close()


    fig, axis = plt.subplots(figsize=(7, 7))
    theta_plot = np.append(theta, np.pi)
    Ea_plot = np.concatenate([Ea, Ea[:1, :]], axis=0)
    EEa = np.max(Ea_plot)
    logEa = np.log10(Ea_plot/EEa).T

    norm = Normalize(vmin=-6.0, vmax=1.0)
    # mesh = axis.pcolormesh(theta_plot, k, logEa, cmap='cmo.dense', norm=norm)

    scale = np.linspace(-6.0, 1, 50)
    cf = axis.contourf(theta_plot, k, logEa, cmap='cmo.dense', levels=scale)
    fig.colorbar(cf, ax=axis, shrink=0.8, format='{x:+.1f}')


    axis.axvline(np.pi/2, color='black', linestyle='--', alpha=0.5)
    axis.set_title(f'Enstrophy Wavelet Spectrum $ \\log  \\tilde z$ | $\\tilde Z = {EEa:.1e}$')
    axis.set_xlabel('Angle')
    axis.set_ylabel('K')
    axis.set_yscale('log')
    axis.set_xlim(0, np.pi)
    fig.tight_layout()
    fig.savefig(res_path / 'z-gamma-th.png', dpi=150)
    plt.show()
    plt.close()

    dax = ds.ux.isel(time=0)
    day = ds.uy.isel(time=0)

    Wax = xwl.cwt2(dax, s, dim=['y', 'x'], x0=x0, ntheta=ntheta, gamma=gamma, wtype='imorlet', center=False)
    Way = xwl.cwt2(day, s, dim=['y', 'x'], x0=x0, ntheta=ntheta, gamma=gamma, wtype='imorlet', center=False)
    Eai = np.abs(Wax)**2 / Wax["s"] * x0**2 + np.abs(Way)**2 / Way["s"] * x0**2

    Wax = xwl.cwt2(dax, s, dim=['y', 'x'], x0=x0, ntheta=ntheta, gamma=gamma, wtype='smorlet', center=False)
    Way = xwl.cwt2(day, s, dim=['y', 'x'], x0=x0, ntheta=ntheta, gamma=gamma, wtype='smorlet', center=False)
    Ea = np.abs(Wax)**2 / Wax["s"] * x0**2 + np.abs(Way)**2 / Way["s"] * x0**2

    k = 2 * np.pi/Ea.s.values
    theta = Ea.angle.values

    Eai = Eai
    Eaz = Ea.sel(angle=0.0, method='nearest')
    Eam = Ea.sel(angle=np.pi/2, method='nearest')
    Eaa = Ea.mean(dim='angle')

    fig, axis = plt.subplots(figsize=(7, 7))
    loglog = axis.loglog(k, Eaz, color=qg.RED, label='zonal (kx)')
    loglog = axis.loglog(k, Eam, color=qg.BLUE, label='meridional (ky)')
    loglog = axis.loglog(k, Eai, color=qg.RED, label='isotropic', linestyle='--')
    loglog = axis.loglog(k, Eaa, color=qg.BLUE, label='averaged', linestyle='--')

    axis.set_title(f'Energy Wavelet Spectrum $\\tilde E$')
    axis.set_xlabel('k')
    axis.set_ylabel('E(k)')
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(res_path / 'E-gamma.png', dpi=150)
    plt.show()
    plt.close()


    fig, axis = plt.subplots(figsize=(7, 7))
    theta_plot = np.append(theta, np.pi)
    Ea_plot = np.concatenate([Ea, Ea[:1, :]], axis=0)
    EEa = np.max(Ea_plot)
    logEa = np.log10(Ea_plot/EEa).T

    norm = Normalize(vmin=-6.0, vmax=1.0)
    # mesh = axis.pcolormesh(theta_plot, k, logEa, cmap='cmo.dense', norm=norm)

    scale = np.linspace(-6.0, 1, 50)
    cf = axis.contourf(theta_plot, k, logEa, cmap='cmo.dense', levels=scale)
    fig.colorbar(cf, ax=axis, shrink=0.8, format='{x:+.1f}')


    axis.axvline(np.pi/2, color='black', linestyle='--', alpha=0.5)
    axis.set_title(f'Energy Wavelet Spectrum $ \\log  \\tilde e$ | $\\tilde E = {EEa:.1e}$')
    axis.set_xlabel('Angle')
    axis.set_ylabel('K')
    axis.set_yscale('log')
    axis.set_xlim(0, np.pi)
    fig.tight_layout()
    fig.savefig(res_path / 'e-gamma-th.png', dpi=150)
    plt.show()
    plt.close()

    return ds

if __name__ == '__main__':
    typer.run(main)