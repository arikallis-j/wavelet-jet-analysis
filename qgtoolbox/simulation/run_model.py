import xarray as xr
from tqdm import tqdm

def epoch_path(path, k_epoch):
    if path is not None:
        return str(path)[:-3:] + f"_{k_epoch}e.nc"
    else:
        return None

def run_epoch(model, path = None):
    traj = []
    if model.t == 0.0:
        traj.append(model.to_dataset())
    for t in tqdm(model.run_with_snapshots(tsnapint=model.twrite), total=round((model.tmax - model.t)/model.twrite)):
        traj.append(model.to_dataset())
    epoch = xr.concat(traj, dim='time', data_vars='all')
    if path is not None:
        epoch.to_netcdf(path, engine="h5netcdf")
    return epoch

def run_simulation(model, n_epoch: int = 1, path = None):
    epochs = []
    t_epoch = model.tmax
    for k in tqdm(range(n_epoch), desc="epoch"):
        epoch = run_epoch(model, path=epoch_path(path, k+1))
        epochs.append(epoch)
        model.tmax += t_epoch
    sim = xr.concat(epochs, dim='time', data_vars='all')
    if path is not None:
        sim.to_netcdf(path, engine="h5netcdf")
    return sim

def load_simulation(path, n_epoch: int = 1):
    epochs = []
    for k in range(n_epoch):
        with xr.open_dataset(epoch_path(path, k+1), engine="h5netcdf") as data:
            epochs.append(data.load())
    return xr.concat(epochs, dim='time', data_vars='all')