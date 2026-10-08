import typer
import qgtoolbox as qg

def main(n_epoch: int = None, k_epoch: int = 1, experiment: str = 'atm', time: int = -1, show: bool = False, fields: str ='qv', size: int = 8, j: int = 1, wtype: str = 'haar'):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    exp_path = path / f"{experiment}"
    if n_epoch is None:
        data_path = exp_path
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    # ds = qg.load_dataset(f"{data_path}.nc")
    ds = qg.load_simulation(f"{exp_path}.nc", n_epoch, k_epoch)
    print(f"time = [{ds.time.values[0]:.0f}, {ds.time.values[-1]:.0f}]")
    ds = qg.calc_diganostic(ds, wtype=wtype)
    snap_path = path / f"{experiment}_t{ds.time[time]:.0f}_{fields}.png"

    fields_keys = list(fields)
    fields_names = []
    kwargs = {
        'ew': {
            'j': j,
        }
    }
    for key in fields_keys:
        fields_names.append(qg.FIELDS[key]) 
    qg.draw_snapshot(ds, time, path=snap_path, colorbar=True, fields=fields_names, show=show, size=size, kwargs=kwargs)
    return ds

if __name__ == '__main__':
    typer.run(main)