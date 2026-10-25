import typer
import qgtoolbox as qg
import matplotlib.pyplot as plt

from tqdm import tqdm

def main(
    n_epoch: int = 1, 
    exp: str = 'atm', 
    method: str ='fourier', 
    n_scale: int = None,
    time: int = None, 
    fields: str ='qUp',
    show: bool = False,
    size: int = 5,
):
    dm = qg.DataManager()
    sim_path = dm.make_dir(path=dm.raw, name=exp)
    res_path = dm.make_dir(path=dm.processed, name=exp)
    stat_path = dm.make_dir(path=res_path, name=method)
    img_path = dm.make_dir(path=res_path, name='img')
    pic_path = img_path / f"{fields}_{n_epoch}e_{str((time if time >= 0 else 100 + time) + 1) + 't' if time is not None else 'mean'}.png"
    cfg_path = dm.config / f"{exp}.yaml"
    ds_stat_path = stat_path / f"{exp}.nc"

    ds_stat_path_e = qg.epoch_path(ds_stat_path, n_epoch)
    ds = qg.load_dataset(ds_stat_path_e)
    if time is None:
        ds = ds.mean(dim='time').expand_dims({'time': [ds.time.values[0]]})
    else:
        ds = ds.isel(time=time).expand_dims({'time': [ds.time.values[time]]})
    
    ds = qg.calc_fourier(ds)
    # ds = qg.calc_wavelets(ds, wtype='imorlet', n_scale=1)
    kwargs = {
        'ew': {
            # 's_idx': -2
        }, 
        # 'ewz': {
        #     's_idx': 0
        # },
        # 'ewm': {
        #     's_idx': 0
        # }
    }
    if show:
        pic_path = None
    qg.draw_snapshot(ds=ds, t_idx=time, path=pic_path, show=show, fields=list(fields), colorbar=False, size=size, kwargs=kwargs)

    return ds

if __name__ == '__main__':
    typer.run(main)