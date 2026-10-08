import typer
import qgtoolbox as qg
import xarray as xr
import numpy as np
import xrft
import xwavelet.wavelet as xwl
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import pywt

from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

from scipy.signal import find_peaks


def main(n_epoch: int = None, experiment: str = 'atm', time: int = -1, show: bool = False, n_jets: int = 3, wtype: str = 'haar'):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    if n_epoch is None:
        n_epoch = 10
        data_path = path / f"{experiment}"
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    
    # ds = qg.load_dataset(f"{data_path}.nc")
    ds = qg.calc_diganostic(ds, n_jets = n_jets, wtype = wtype)
    # print(ds.time)
    # ds = ds.isel(time=[time])
    # print(ds)
    # dx = ds.x.values[1] - ds.x.values[0]
    # n_grid = ds.attrs['grid.n_grid']
    # levels = ds.attrs['grid.n_lev']

    # x = ds.x.values
    # y = ds.y.values

    # n_jets = 3
    # x0_idx = 128

    # V = np.abs(ds.ux).max().values
    # vx = ds.ux.isel(time=time, x=x0_idx).values / V

    # vx_periodic = np.concatenate([vx, vx])
    # peaks_idx, _ = find_peaks(vx_periodic, distance=n_grid//(2*n_jets))
    # peaks_idx = peaks_idx[peaks_idx < n_grid]
    # y_jets_idx = peaks_idx[np.argsort(vx[peaks_idx])[::-1][:n_jets]]
    # print(y_jets_idx)

    # avx_periodic = np.concatenate([-vx, -vx])
    # peaks_idx, _ = find_peaks(avx_periodic, distance=n_grid//(2*n_jets))
    # peaks_idx = peaks_idx[peaks_idx < n_grid]
    # y_ajets_idx = peaks_idx[np.argsort(-vx[peaks_idx])[::-1][:n_jets]]
    # print(y_ajets_idx)

    # fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # for i, idx in enumerate(y_jets_idx):
    #     ew_jet = ds.ew.isel(time=time, x=x0_idx, y=idx).values
    #     k_jet = n_grid / ds.s.values 
    #     axes[0, i].loglog(k_jet, ew_jet)
    #     axes[0, i].set_title(f'Jet y={y[idx]:.2f}')

    # for i, idx in enumerate(y_ajets_idx):
    #     ew_ajet = ds.ew.isel(time=time, x=x0_idx, y=idx).values
    #     k_jet = n_grid / ds.s.values 
    #     axes[1, i].loglog(k_jet, ew_ajet)
    #     axes[1, i].set_title(f'Anti-jet y={y[idx]:.2f}')

    # plt.show()

    # x0_idx = 128 + 1

    # y_jet = ds.y_jet.isel(time=time)   # (n_jets, n_x)
    # y_ajet = ds.y_ajet.isel(time=time)

    # ew = ds.ew.isel(time=time)  # (n_scales, n_y, n_x)

    # ew_jets = ew.sel(y=y_jet, method='nearest')   # (n_scales, n_jets, n_x)
    # ew_ajets = ew.sel(y=y_ajet, method='nearest')

    # ew_jets_mean = ew_jets.mean(dim=['x','k_jet'])  # (n_scales, n_jets)
    # ew_ajets_mean = ew_ajets.mean(dim=['x','k_jet'])

    # print(ew_jets)
    # print(ew_jets_mean)

    # fig, axis = plt.subplots(figsize=(8, 8))

    # k_freq = n_grid / ds.s.values 
    # axis.loglog(k_freq, ew_jets_mean.values, label='Jet', color=qg.BLUE)
    # axis.loglog(k_freq, ew_ajets_mean.values, label='Anti-jet', color=qg.RED)
    # axis.legend()
    # plt.show()

    # # x0 = ds.x.values[1] - ds.x.values[0]
    # # levels = xwl.swt_max_level(n_grid)
    # # wtype = 'haar'


    # # Wax = xwl.swt2(ds.ux, levels=levels, wtype=wtype)
    # # Eax = Wax['cH']**2 + Wax['cV']**2 + Wax['cD']**2

    # # Way = xwl.swt2(ds.uy, levels=levels, wtype=wtype)
    # # Eay = Way['cH']**2 + Way['cV']**2 + Way['cD']**2

    # # Ea = Eax + Eay
    # # # print(Ea)

    # # AHa = (Wax['cH']**2 + Way['cH']**2) / Ea
    # # # AVa = (Wax['cV']**2 + Way['cV']**2) / Ea
    # # # ADa = (Wax['cD']**2 + Way['cD']**2) / Ea
    
    # # Ea = ds.ew.isel(time=time)
    # # gamma_x, gamma_y = Ea["x"], Ea["y"]

    # # qg.draw_snapshot(ds, time, path = path / "img", show=show, fields=['ew'])

    # # for j in range(1, levels+1):
    # #     Ea_j = Ea.isel(s=j-1)
    # #     k = n_grid / Ea_j["s"]
    # #     eh = Ea_j.values
    # #     Eh = np.abs(Ea_j.values).max()
    # #     norm = LogNorm(vmin=1e-6, vmax=1e1)

    # #     fig, axis = plt.subplots()
    # #     fig.suptitle(f"Time t = {Ea_j.time:.1f} | Octave j = {j}")
    # #     mesh = qg.plot_swt_field(ds, axis, t_idx=time, j=j)
    # #     fig.colorbar(mesh, ax=axis)
    # #     fig.tight_layout()
    # #     if path is not None:
    # #         fig.savefig( path / "img" / f"e{j}", dpi=150)
    # #     if show:
    # #         plt.show()
    # #     plt.close()

    # # AHa = AHa.isel(time=time)
    # # gamma_x, gamma_y = AHa["x"], AHa["y"]

    # # for j in range(1, levels+1):
    # #     Ea_j = AHa.isel(s=j-1)
    # #     k = 2 * np.pi / (Ea_j["s"] * x0)
    # #     eh = Ea_j.values
    # #     Eh = 1.0
    # #     norm = Normalize(vmin=0.0, vmax=1.0)

    # #     fig, axis = plt.subplots()
    # #     fig.suptitle(f"Time t = {Ea_j.time:.1f} | Octave j = {j}")
    # #     mesh = axis.pcolormesh(gamma_x, gamma_y, eh/Eh, cmap='cmo.deep_r', norm=norm)
    # #     axis.set_title(f'Anisotropy Horizontal $e_j(x, y)$ | $ E_j = {Eh:.1e}, k_j = {k.values:.0f}$')
    # #     axis.set_aspect('equal')
    # #     axis.set_xlabel('X')
    # #     axis.set_ylabel('Y')
    # #     fig.colorbar(mesh, ax=axis)
    # #     fig.tight_layout()
    # #     if path is not None:
    # #         fig.savefig( path / "img" / f"ah{j}", dpi=150)
    # #     if show:
    # #         plt.show()
    # #     plt.close()

    # # AVa = AVa.isel(time=time)
    # # gamma_x, gamma_y = AVa["x"], AVa["y"]

    # # for j in range(1, levels+1):
    # #     Ea_j = AVa.isel(s=j-1)
    # #     k = 2 * np.pi / (Ea_j["s"] * x0)
    # #     eh = Ea_j.values
    # #     Eh = 1.0
    # #     norm = Normalize(vmin=0.0, vmax=1.0)

    # #     fig, axis = plt.subplots()
    # #     fig.suptitle(f"Time t = {Ea_j.time:.1f} | Octave j = {j}")
    # #     mesh = axis.pcolormesh(gamma_x, gamma_y, eh/Eh, cmap='cmo.deep_r', norm=norm)
    # #     axis.set_title(f'Anisotropy Vertical $e_j(x, y)$ | $ E_j = {Eh:.1e}, k_j = {k.values:.0f}$')
    # #     axis.set_aspect('equal')
    # #     axis.set_xlabel('X')
    # #     axis.set_ylabel('Y')
    # #     fig.colorbar(mesh, ax=axis)
    # #     fig.tight_layout()
    # #     if path is not None:
    # #         fig.savefig( path / "img" / f"av{j}", dpi=150)
    # #     if show:
    # #         plt.show()
    # #     plt.close()


    # # ADa = ADa.isel(time=time)
    # # gamma_x, gamma_y = ADa["x"], ADa["y"]

    # # for j in range(1, levels+1):
    # #     Ea_j = ADa.isel(s=j-1)
    # #     k = 2 * np.pi / (Ea_j["s"] * x0)
    # #     eh = Ea_j.values
    # #     Eh = 1.0
    # #     norm = Normalize(vmin=0.0, vmax=1.0)

    # #     fig, axis = plt.subplots()
    # #     fig.suptitle(f"Time t = {Ea_j.time:.1f} | Octave j = {j}")
    # #     mesh = axis.pcolormesh(gamma_x, gamma_y, eh/Eh, cmap='cmo.deep_r', norm=norm)
    # #     axis.set_title(f'Anisotropy Diagonal $e_j(x, y)$ | $ E_j = {Eh:.1e}, k_j = {k.values:.0f}$')
    # #     axis.set_aspect('equal')
    # #     axis.set_xlabel('X')
    # #     axis.set_ylabel('Y')
    # #     fig.colorbar(mesh, ax=axis)
    # #     fig.tight_layout()
    # #     if path is not None:
    # #         fig.savefig( path / "img" / f"ad{j}", dpi=150)
    # #     if show:
    # #         plt.show()
    # #     plt.close()


    return ds

if __name__ == '__main__':
    typer.run(main)