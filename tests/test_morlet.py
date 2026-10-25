import typer
import qgtoolbox as qg
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xwavelet.wavelet as xwl
import xrft

from tqdm import tqdm

def plot_imorlet2_map(ds, s_idx, gamma, path):
    x, y = ds.x, ds.y
    a = 1.0
    n_grid = ds.attrs['grid.n_grid']
    n_scale = n_grid
    x0 = ds.x.values[1] - ds.x.values[0]
    s_vals = np.geomspace(2.0, n_grid, n_scale)
    s = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})

    imorlet2, phi = xwl._iso_morlet2(x0, a, s, y, x, gamma)
    imorlet2 = imorlet2.isel(s=s_idx) 
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f"2D IsoMorlet | s = {imorlet2.s.values:0.1f} | gamma = ({gamma[0]/(np.pi):0.1f}$\\pi$, {gamma[1]/(np.pi):0.1f}$\\pi$)")

    axis = axes[0]
    mesh = axis.pcolormesh(x, y, imorlet2.real, vmin=-1.0, vmax=1.0)
    axis.set_title(f'Real')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    axis = axes[1]
    mesh = axis.pcolormesh(x, y, imorlet2.imag, vmin=-1.0, vmax=1.0)
    axis.set_title(f'Imag')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    axis = axes[2]
    mesh = axis.pcolormesh(x, y, np.abs(imorlet2), vmin=0.0, vmax=1.0)
    axis.set_title(f'Abs')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()

def plot_delta_morlet2_map(ds, s_idx, th_idx, gamma, path):
    x, y = ds.x, ds.y
    ntheta = 50
    a = 1.0
    n_grid = ds.attrs['grid.n_grid']
    n_scale = n_grid
    x0 = ds.x.values[1] - ds.x.values[0]
    s_vals = np.geomspace(2.0, n_grid, n_scale)
    s = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})
    s = s.isel(s=[s_idx])

    # morlet2, phi = xwl._morlet2(x0, ntheta, a, s, y, x)
    imorlet2, phi = xwl._iso_morlet2(x0, a, s, y, x, gamma)
    imorlet2 = imorlet2.isel(s=0) 

    morlet2, phi = xwl._shift_morlet2(x0, ntheta, a, s, y, x, gamma)
    print(phi.values/(np.pi)*180)
    morlet2 = morlet2.mean(dim='angle')
    morlet2 = morlet2.isel(s=0) 
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    # fig.suptitle(f"2D Morlet | s = {morlet2.s.values:0.1f} | theta = {morlet2.angle.values / (np.pi)*180 :0.1f} | gamma = ({gamma[0]/(np.pi):0.1f}$\\pi$, {gamma[1]/(np.pi):0.1f}$\\pi$)")
    fig.suptitle(f"2D Morlet | s = {morlet2.s.values:0.1f}")


    axis = axes[0]
    mesh = axis.pcolormesh(x, y, morlet2.real - imorlet2.real)
    axis.set_title(f'Real')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    axis = axes[1]
    mesh = axis.pcolormesh(x, y, morlet2.imag - imorlet2.imag)
    axis.set_title(f'Imag')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    axis = axes[2]
    mesh = axis.pcolormesh(x, y, np.abs(morlet2) - np.abs(imorlet2))
    axis.set_title(f'Abs')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    fig.colorbar(mesh, ax=axis, shrink=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


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

    # morlet2, phi = xwl._morlet2(x0, ntheta, a, s, y, x)
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

def plot_morlet2(ds, s_idx, path):
    gamma = (0, 0)
    x = ds.x - ds.x.mean()
    # y = 0.0
    y = ds.y


    ntheta = 2
    a = 1.0
    n_grid = ds.attrs['grid.n_grid']
    n_scale = n_grid
    x0 = ds.x.values[1] - ds.x.values[0]
    s_vals = np.geomspace(2.0, n_grid, n_scale)
    s = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})
    # sdim = s.dims[0]

    # morlet2, phi = xwl._morlet2(x0, ntheta, a, s, y, x)
    morlet2, phi = xwl._shift_morlet2(x0, ntheta, a, s, y, x, gamma)
    # morlet2 = morlet2.isel(angle=0)
    morlet2 = morlet2.isel(angle=0).isel(y=0)
    morlet2 = morlet2.isel(s=s_idx) 
    exp2 = np.exp(-1j * x * n_grid/morlet2.s.values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"2D Morlet | s = {morlet2.s.values:0.1f}")

    axis = axes[0]
    mesh = axis.plot(x, morlet2.real, label='morlet')
    mesh = axis.plot(x, exp2.real, label='exp')
    axis.set_title(f'Real')
    axis.set_xlabel('X')
    axis.set_ylabel('F(X)')
    axis.legend()

    axis = axes[1]
    mesh = axis.plot(x, morlet2.imag, label='morlet')
    mesh = axis.plot(x, exp2.imag, label='exp')
    axis.set_title(f'Imag')
    axis.set_xlabel('X')
    axis.set_ylabel('F(X)')
    axis.legend()
    fig.tight_layout()

    fig.savefig(path, dpi=150)
    plt.show()


def main(n_epoch: int = 1, exp: str = 'test', s_idx: int = 1):
    dm = qg.DataManager()
    sim_path = dm.make_dir(path=dm.raw, name=exp)
    res_path = dm.make_dir(path=dm.processed, name=exp)
    ds_sim_path = sim_path / f"{exp}.nc"
    ds = qg.load_dataset(ds_sim_path)
    ds = qg.calc_statistics(ds)
    # plot_morlet2(ds, s_idx=s_idx, path=res_path / f'smorlet2-r_{s_idx}.png')
    th_idx = 0
    gamma = (np.pi, np.pi)
    plot_delta_morlet2_map(ds, s_idx=s_idx, th_idx=th_idx, gamma=gamma, path=res_path / f'delta-morlet2-iso.png')
    plot_morlet2_map(ds, s_idx=s_idx, th_idx=None, gamma=gamma, path=res_path / f'smorlet2-mean.png')
    plot_imorlet2_map(ds, s_idx=s_idx, gamma=gamma, path=res_path / f'imorlet2.png')
    plot_morlet2_map(ds, s_idx=s_idx, th_idx=th_idx, gamma=gamma, path=res_path / f'smorlet2.png')


    return ds

if __name__ == '__main__':
    typer.run(main)