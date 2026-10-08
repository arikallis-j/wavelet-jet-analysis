import xarray as xr
from tqdm import tqdm

def epoch_path(path, k_epoch):
    if path is not None:
        return str(path)[:-3:] + f"_{k_epoch}e.nc"
    else:
        return None

def make_dataset(model, config = None, data_vars = ['q','ux','uy','p']):
    ds = model.to_dataset() 
    ds = ds.rename({'u': 'ux', 'v': 'uy'})
    ds = ds[data_vars].isel(lev=0)
    new_attrs = {}
    if config is not None:
        for param, val in config.items():
            if not isinstance(val, dict):
                new_attrs[param] = val
            else:
                for key, val_deep in val.items():
                    new_attrs[f"{param}.{key}"] = val_deep
    ds.attrs = new_attrs
    return ds

def run_epoch(model, path = None, config = None):
    traj = []
    if model.t == 0.0:
        traj.append(make_dataset(model, config=config))
    for t in tqdm(model.run_with_snapshots(tsnapint=model.twrite), total=round((model.tmax - model.t)/model.twrite)):
        traj.append(make_dataset(model, config=config))
    epoch = xr.concat(traj, dim='time', data_vars='all')
    if path is not None:
        epoch.to_netcdf(path, engine="h5netcdf")
    return epoch

def run_simulation(model, n_epoch: int = 1, path = None, config = None, save_mode = 'a'):
    epochs = []
    t_epoch = model.tmax
    for k in tqdm(range(n_epoch), desc="epoch"):
        epoch = run_epoch(model, path=epoch_path(path, k+1), config=config)
        epochs.append(epoch)
        model.tmax += t_epoch
    if save_mode == 'a':
        sim = xr.concat(epochs, dim='time', data_vars='all')
        if path is not None:
            sim.to_netcdf(path, engine="h5netcdf")
        return sim
    elif save_mode == 'e':
        return epochs
    else:
        return None

def load_simulation(path, n_epoch: int = 1, k_epoch: int = 1):
    epochs = []
    for k in range(k_epoch, k_epoch+n_epoch):
        with xr.open_dataset(epoch_path(path, k), engine="h5netcdf") as data:
            epochs.append(data.load())
    return xr.concat(epochs, dim='time', data_vars='all')