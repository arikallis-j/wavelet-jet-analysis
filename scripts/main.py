import typer
import qgtoolbox as qg
import xarray as xr
import numpy as np
import xrft
import xwavelet.wavelet as xwl # cwt, cwt2, power_spectrum, cross_spectrum
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import cmocean.cm as cmo

from matplotlib.colors import LogNorm
from tqdm import tqdm

# Ea = xwl.power_spectrum(da, scale, dim=['y', 'x'], x0=x0, ntheta=ntheta, normalize=False)

def main(n_epoch: int = None, experiment: str = 'atm', time: int = -1, show: bool = False, nwl: int = 10):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    if n_epoch is None:
        data_path = path / f"{experiment}"
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    ds = qg.load_dataset(f"{data_path}.nc")
    ds = qg.calc_diganostic(ds)

    ds = ds.isel(time=time)

    n_scale, n_theta, n_gamma = nwl, 2 * nwl, nwl
    n_grid = ds.attrs['grid.n_grid']
    dl = ds.x.values[1] - ds.x.values[0]
    L = 2*np.pi 
    x0 = dl
    s_vals = 1 / np.linspace(1/n_grid, 1, n_scale)
    s_vals = np.geomspace(2.0, n_grid, n_scale)
    scale = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})
    gamma_0 = np.array([-ds.x.mean().values, -ds.y.mean().values])

    # da = ds.q.isel(time=time)
    # Wa = xwl.cwt2(da, scale, dim=['y', 'x'], x0=x0, gamma=gamma, ntheta=n_theta, wtype=wtype)
    # Ea = np.abs(Wa)**2 / Wa["s"] * x0**2

    # wtype = 'smorlet'

    # gamma_x, gamma_y = L/2, L/2
    # gamma_0 = np.array([-ds.x.mean().values, -ds.y.mean().values])
    # gamma = np.array([gamma_x, gamma_y]) + gamma_0
    
    # dax = ds.ux
    # Wax = xwl.cwt2(dax, scale, dim=['y', 'x'], x0=x0, gamma=gamma, ntheta=n_theta, wtype=wtype)
    # Eax = np.abs(Wax)**2 / Wax["s"] * x0**2
    
    # day = ds.uy
    # Way = xwl.cwt2(day, scale, dim=['y', 'x'], x0=x0, gamma=gamma, ntheta=n_theta, wtype=wtype)
    # Eay = np.abs(Way)**2 / Way["s"] * x0**2

    # Ea = Eax + Eay

    # k = 2 * np.pi / Ea["s"]
    # theta = Ea["angle"]
    # azimut = theta - np.pi/2

    # eh = Ea.T.values
    # Eh = np.abs(Ea.values).max()
    # norm = LogNorm(vmin=1e-6, vmax=1e1)
    # plt.pcolormesh(azimut, k, eh/Eh, cmap='cmo.dense', norm=norm)
    # plt.yscale('log')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    wtype = 'imorlet'

    gamma_x_vals = np.linspace(0, L, n_gamma, endpoint=False)
    gamma_y_vals = np.linspace(0, L, n_gamma, endpoint=False)

    Ea_x_list = []
    for gamma_x in tqdm(gamma_x_vals):
        Ea_y_list = []
        for gamma_y in gamma_y_vals:
            gamma = np.array([gamma_x, gamma_y]) + gamma_0

            Wax = xwl.cwt2(ds.ux, scale, dim=['y', 'x'], x0=x0, gamma=gamma, wtype=wtype)
            Eax = np.abs(Wax)**2 / Wax["s"] * x0**2

            Way = xwl.cwt2(ds.uy, scale, dim=['y', 'x'], x0=x0, gamma=gamma, wtype=wtype)
            Eay = np.abs(Way)**2 / Way["s"] * x0**2

            Ea_y_list.append(Eax + Eay)

        Ea_x_list.append(xr.concat(Ea_y_list, dim=xr.DataArray(gamma_y_vals, dims=["gamma_y"])))

    Ea = xr.concat(Ea_x_list, dim=xr.DataArray(gamma_x_vals, dims=["gamma_x"]))
    
    print(Ea)
    Ea = Ea.mean(dim="gamma_x")

    k = 2 * np.pi / Ea["s"]
    gamma_y = Ea["gamma_y"]

    eh = Ea.T.values
    Eh = np.abs(Ea.values).max()
    norm = LogNorm(vmin=1e-6, vmax=1e1)
    plt.pcolormesh(gamma_y, k, eh/Eh, cmap='cmo.dense', norm=norm)
    plt.yscale('log')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return ds

if __name__ == '__main__':
    typer.run(main)