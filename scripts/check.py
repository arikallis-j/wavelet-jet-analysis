import typer
import qgtoolbox as qg
import numpy as np
import xrft

def calc_diganostic(ds):
    ds['v'] = np.sqrt(ds['ux']**2 + ds['uy']**2)
    ds['E'] = xrft.isotropic_power_spectrum(
        ds['ux'] + 1j*ds['uy'], dim=['x', 'y'], nfactor=4, truncate=True
    )
    return ds

def main(n_epoch: int = None, experiment: str = 'atm', time: int = -1, show: bool = False, fields: str ='qv'):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    if n_epoch is None:
        data_path = path / f"{experiment}"
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    ds = qg.load_dataset(f"{data_path}.nc")
    ds = calc_diganostic(ds)
    ds.info()
    snap_path = path / f"{experiment}_t{ds.time[time]:.0f}.png"
    qg.draw_snapshot(ds, time, path=snap_path, colorbar=True, fields=list(fields), show=show)
    return ds

if __name__ == '__main__':
    typer.run(main)