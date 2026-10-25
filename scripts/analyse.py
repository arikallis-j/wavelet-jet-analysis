import typer
import qgtoolbox as qg
import matplotlib.pyplot as plt

from tqdm import tqdm

def main(n_epoch: int = 1, exp: str = 'atm', method='fourier'):
    dm = qg.DataManager()
    sim_path = dm.make_dir(path=dm.raw, name=exp)
    res_path = dm.make_dir(path=dm.processed, name=exp)
    stat_path = dm.make_dir(path=res_path, name=method)
    cfg_path = dm.config / f"{exp}.yaml"
    ds_sim_path = sim_path / f"{exp}.nc"
    ds_stat_path = stat_path / f"{exp}.nc"

    for epoch in tqdm(range(1, n_epoch + 1, -1)):
        ds_sim_path_e = qg.epoch_path(ds_sim_path, epoch)
        ds_stat_path_e = qg.epoch_path(ds_stat_path, epoch)
        ds = qg.load_dataset(ds_sim_path_e)
        ds = qg.calc_statistics(ds)
        if method == 'fourier':
            ds = qg.calc_fourier(ds)
        else:
            raise Warning(f"There is no implementation of {method} method")
        qg.save_dataset(data=ds, path=ds_stat_path_e)

    return ds

if __name__ == '__main__':
    typer.run(main)