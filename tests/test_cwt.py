import typer
import qgtoolbox as qg
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xwavelet.wavelet as xwl
import xrft

from tqdm import tqdm
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

def compute_jet_coords(ds, dims=['y', 'x'], n_jets=3):
    n_grid = ds.attrs['grid.n_grid']
    n_times = len(ds.time)
    x = ds.x.values
    y = ds.y.values

    y_jet_arr = np.zeros((n_times, n_jets, n_grid), dtype=float)
    y_ajet_arr = np.zeros((n_times, n_jets, n_grid), dtype=float)
    y_njet_arr = np.zeros((n_times, n_jets, n_grid), dtype=float)
    V = np.abs(ds.ux).max().values

    for t in range(n_times):
        for xi in range(n_grid):
            vx = ds.ux.isel(time=t, x=xi).values / V
            vx_periodic = np.concatenate([vx, vx])
            peaks_idx, _ = find_peaks(vx_periodic, distance=n_grid//(1.5*n_jets))
            peaks_idx = peaks_idx[peaks_idx < n_grid]
            top = peaks_idx[np.argsort(vx[peaks_idx])[::-1][:n_jets]]
            if len(top) < n_jets:
                top = np.pad(top, (0, n_jets - len(top)), constant_values=-1)
            jet_idx = np.sort(top)
            y_jet_arr[t, :, xi] = np.where(jet_idx == -1, np.nan, y[np.clip(jet_idx, 0, len(y)-1)])

            avx = -vx
            avx_periodic = np.concatenate([avx, avx])
            peaks_idx, _ = find_peaks(avx_periodic, distance=n_grid//(1.5*n_jets))
            peaks_idx = peaks_idx[peaks_idx < n_grid]
            atop = peaks_idx[np.argsort(avx[peaks_idx])[::-1][:n_jets]]
            if len(atop) < n_jets:
                atop = np.pad(atop, (0, n_jets - len(atop)), constant_values=-1)
            ajet_idx = np.sort(atop)
            y_ajet_arr[t, :, xi] = np.where(ajet_idx == -1, np.nan, y[np.clip(ajet_idx, 0, len(y)-1)])
        
            nvx = 1/np.abs(vx)
            nvx_periodic = np.concatenate([nvx, nvx])
            peaks_idx, _ = find_peaks(nvx_periodic, distance=n_grid//(1.5*n_jets))
            peaks_idx = peaks_idx[peaks_idx < n_grid]
            ntop = peaks_idx[np.argsort(nvx[peaks_idx])[::-1][:n_jets]]
            if len(ntop) < n_jets:
                ntop = np.pad(ntop, (0, n_jets - len(ntop)), constant_values=-1)
            njet_idx = np.sort(ntop)
            y_njet_arr[t, :, xi] = np.where(njet_idx == -1, np.nan, y[np.clip(njet_idx, 0, len(y)-1)])

    y_jet = xr.DataArray(
        y_jet_arr,
        dims=['time', 'k_jet', 'x'],
        coords={'k_jet': np.arange(1, n_jets+1)},
        attrs={'n_jet': n_jets}
    )
    y_ajet = xr.DataArray(
        y_ajet_arr,
        dims=['time', 'k_jet', 'x'],
        coords={'k_jet': np.arange(1, n_jets+1)},
        attrs={'n_jet': n_jets}
    )
    y_njet = xr.DataArray(
        y_njet_arr,
        dims=['time', 'k_jet', 'x'],
        coords={'k_jet': np.arange(1, n_jets+1)},
        attrs={'n_jet': n_jets}
    )
    return y_jet, y_ajet, y_njet

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
    # sigma = 10  # в пикселях, подбираешь руками
    # q_smooth = gaussian_filter(ds.q.values, sigma=sigma)
    # q_smooth = xr.DataArray(q_smooth, dims=ds.q.dims, coords=ds.q.coords)
    # ds['q'] = q_smooth

    fig, axis = plt.subplots(figsize=(7, 7))
    qg.plot_zonal_velocity(ds, axis, t_idx=0)
    axis.axvline(gamma[0], color='grey', linestyle='--', alpha=0.5)
    axis.axhline(gamma[1], color='grey', linestyle='--', alpha=0.5)
    axis.scatter(gamma[0], gamma[1], color='black', marker='+', s=1000)
    fig.tight_layout()
    fig.savefig(path / 'ux-gamma.png', dpi=150)
    plt.show()
    plt.close()

    fig, axis = plt.subplots(figsize=(7, 7))
    qg.plot_relative_vorticity(ds, axis, t_idx=0)
    axis.axvline(gamma[0], color='grey', linestyle='--', alpha=0.5)
    axis.axhline(gamma[1], color='grey', linestyle='--', alpha=0.5)
    axis.scatter(gamma[0], gamma[1], color='black', marker='+', s=1000)
    fig.tight_layout()
    fig.savefig(path / 'q-gamma.png', dpi=150)
    plt.show()
    plt.close()

def calc_spectra(ds, gamma=None, gamma_idx=None, n_scale=10, direction='x'):
    x, y = ds.x, ds.y  
    n_grid = ds.attrs['grid.n_grid']
    x0 = ds.x.values[1] - ds.x.values[0]
    s_vals = np.geomspace(2.0, n_grid, n_grid)
    s = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})
    # Wax = xwl.cwt2(ds.ux, s, dim=['y', 'x'], x0=x0, ntheta=4, gamma=gamma, wtype='smorlet', center=False)
    # Way = xwl.cwt2(ds.uy, s, dim=['y', 'x'], x0=x0, ntheta=4, gamma=gamma, wtype='smorlet', center=False)
    Wax = xwl.cwt2_field(ds.ux, s, dim=['y', 'x'], x0=x0, ntheta=4, n_gamma=n_scale, gamma_idx=gamma_idx, wtype='smorlet', direction=direction)
    Way = xwl.cwt2_field(ds.uy, s, dim=['y', 'x'], x0=x0, ntheta=4, n_gamma=n_scale, gamma_idx=gamma_idx, wtype='smorlet', direction=direction)
    Ea = np.abs(Wax)**2 / Wax["s"] * x0**2 + np.abs(Way)**2 / Way["s"] * x0**2
    return Ea

def main(n_epoch: int = 1, exp: str = 'test', n_scale: int = 10):
    dm = qg.DataManager()
    sim_path = dm.make_dir(path=dm.raw, name=exp)
    res_path = dm.make_dir(path=dm.processed, name=exp)
    ds_sim_path = sim_path / f"{exp}.nc"
    ds = qg.load_dataset(ds_sim_path)
    ds = ds.mean(dim='time').expand_dims(dim="time", axis=0)
    ds = qg.calc_statistics(ds)
    th_idx = 0
    direction = 'x'
    Ea = []
    # gamma_idx = (92, 215) # njet
    # gamma_idx = (92, 135) # ajet
    # gamma_idx = (92, 93) # jet
    # gamma_idx = (92, 90)
    # gamma_idx = (92, 140)
    n_gamma = 5
    points = np.round(np.linspace(0, len(ds.y.values) - 1, n_gamma+1)).astype(int)[:n_gamma:]


    for k in tqdm(range(n_gamma)):
        gamma_idx = (points[k], 93)
        gamma = (ds.x.isel(x=gamma_idx[0]).values, ds.y.isel(y=gamma_idx[1]).values)
        # plot_gamma(ds, gamma, res_path)
        g_idx = gamma_idx[0] if direction=='x' else gamma_idx[1]

        Ea_k = calc_spectra(ds, gamma_idx=g_idx, n_scale=n_scale, direction=direction)
        Ea.append(Ea_k)
    Ea = xr.concat(Ea, dim='x', data_vars='all')
    Ea = Ea.mean(dim='x').expand_dims(dim="x", axis=0)
    print(Ea)
    Ea = Ea.isel(time=0)/n_gamma
    Eaz = Ea.sel(angle=0.0, method='nearest')
    Eam = Ea.sel(angle=np.pi/2, method='nearest')
    k = 2 * np.pi/Ea.s.values
    theta = Ea.angle.values
    print(Ea)

    fig, axis = plt.subplots(figsize=(7, 7))
    qg.plot_zonal_velocity(ds, axis, t_idx=0)

    if direction=='y':
        axis.axhline(Ea.y.values, color='grey', linestyle='--', alpha=0.5)
        for x in Ea.x.values:
            axis.scatter(x, Ea.y.values, color='black', marker='+', s=100)
    else:
        axis.axvline(Ea.x.values, color='grey', linestyle='--', alpha=0.5)
        for y in Ea.y.values:
            axis.scatter(Ea.x.values, y, color='black', marker='+', s=100)
    fig.tight_layout()
    fig.savefig(res_path / 'ux-gammas.png', dpi=150)
    # plt.show()
    plt.close()

    if direction=='y':
        x, y = Ea.x, Ea.y
        Ez = Eaz.isel(y=0)
        Em = Eam.isel(y=0)

        EEz = np.max(Ez)
        EEm = np.max(Em)
        logEz = np.log10(Ez/EEz).T
        logEm = np.log10(Em/EEm).T
        norm = Normalize(vmin=-6.0, vmax=1.0)
        
        fig, axis = plt.subplots(figsize=(7, 7))
        mesh = axis.pcolormesh(x, k, logEz, cmap='cmo.dense', norm=norm)
        fig.colorbar(mesh, ax=axis, shrink=0.8, format='{x:+.1f}')
        axis.set_title(f'Zonal Wavelet Spectrum $ \\log  \\tilde e_z$ | $\\tilde E_z = {EEz:.1e}$')
        axis.set_xlabel('X')
        axis.set_ylabel('K')
        axis.set_yscale('log')
        axis.set_xlim(0, 2*np.pi)
        fig.tight_layout()
        fig.savefig(res_path / 'ez-gammas.png', dpi=150)
        # plt.show()
        plt.close()

        fig, axis = plt.subplots(figsize=(7, 7))
        mesh = axis.pcolormesh(x, k, logEm, cmap='cmo.dense', norm=norm)
        fig.colorbar(mesh, ax=axis, shrink=0.8, format='{x:+.1f}')
        axis.set_title(f'Meridional Wavelet Spectrum $ \\log  \\tilde e_m$ | $\\tilde E_m = {EEm:.1e}$')
        axis.set_xlabel('X')
        axis.set_ylabel('K')
        axis.set_yscale('log')
        axis.set_xlim(0, 2*np.pi)
        fig.tight_layout()
        fig.savefig(res_path / 'em-gammas.png', dpi=150)
        # plt.show()
        plt.close()

        Eaz = Ez.mean(dim='x')
        Eam = Em.mean(dim='x')
        EEa = np.max(np.abs(Eam))

        fig, axis = plt.subplots(figsize=(7, 7))
        loglog = axis.loglog(k, Eam/EEa, color=qg.BLUE, label='meridional (ky)')
        loglog = axis.loglog(k, Eaz/EEa, color=qg.RED, label='zonal (kx)')
        axis.set_title(f'Energy Wavelet Spectrum $\\tilde E$ | $\\tilde E_0 = {EEa:.1e}$')
        axis.set_xlabel('k')
        axis.set_ylabel('E(k)')
        axis.set_ylim(1e-8, 1e1)
        axis.grid(True)
        axis.legend()
        fig.tight_layout()
        fig.savefig(res_path / 'E-gammas.png', dpi=150)
        # plt.show()
        plt.close()

    else:
        x, y = Ea.x, Ea.y
        Ez = Eaz.isel(x=0)
        Em = Eam.isel(x=0)

        EEz = np.max(Ez)
        EEm = np.max(Em)
        logEz = np.log10(Ez/EEz)
        logEm = np.log10(Em/EEm)
        norm = Normalize(vmin=-6.0, vmax=1.0)
        
        fig, axis = plt.subplots(figsize=(7, 7))
        mesh = axis.pcolormesh(k, y, logEz, cmap='cmo.dense', norm=norm)
        fig.colorbar(mesh, ax=axis, shrink=0.8, format='{x:+.1f}')
        axis.set_title(f'Zonal Wavelet Spectrum $ \\log  \\tilde e_z$ | $\\tilde E_z = {EEz:.1e}$')
        axis.set_xlabel('K')
        axis.set_ylabel('Y')
        axis.set_xscale('log')
        axis.set_ylim(0, 2*np.pi)
        fig.tight_layout()
        fig.savefig(res_path / 'ez-gammas.png', dpi=150)
        # plt.show()
        plt.close()

        fig, axis = plt.subplots(figsize=(7, 7))
        mesh = axis.pcolormesh(k, y, logEm, cmap='cmo.dense', norm=norm)
        fig.colorbar(mesh, ax=axis, shrink=0.8, format='{x:+.1f}')
        axis.set_title(f'Meridional Wavelet Spectrum $ \\log  \\tilde e_m$ | $\\tilde E_m = {EEm:.1e}$')
        axis.set_xlabel('K')
        axis.set_ylabel('Y')
        axis.set_xscale('log')
        axis.set_ylim(0, 2*np.pi)
        fig.tight_layout()
        fig.savefig(res_path / 'em-gammas.png', dpi=150)
        # plt.show()
        plt.close()

        Eaz = Ez.mean(dim='y')
        Eam = Em.mean(dim='y')
        EEa = np.max(np.abs(Eam))

        fig, axis = plt.subplots(figsize=(7, 7))
        loglog = axis.loglog(k, Eam/EEa, color=qg.BLUE, label='meridional (ky)')
        loglog = axis.loglog(k, Eaz/EEa, color=qg.RED, label='zonal (kx)')
        axis.set_title(f'Energy Wavelet Spectrum $\\tilde E$ | $\\tilde E_0 = {EEa:.1e}$')
        axis.set_xlabel('k')
        axis.set_ylabel('E(k)')
        axis.set_ylim(1e-8, 1e1)
        axis.grid(True)
        axis.legend()
        fig.tight_layout()
        fig.savefig(res_path / 'E-gammas.png', dpi=150)
        # plt.show()
        plt.close()


    return ds

if __name__ == '__main__':
    typer.run(main)