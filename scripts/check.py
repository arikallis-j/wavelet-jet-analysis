import typer
import qgtoolbox as qg

def main(n_epoch: int = None, experiment: str = 'atm', time: int = -1, show: bool = False, fields: str ='qv', size: int = 8):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    if n_epoch is None:
        data_path = path / f"{experiment}"
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    ds = qg.load_dataset(f"{data_path}.nc")
    ds = qg.calc_diganostic(ds)
    snap_path = path / f"{experiment}_t{ds.time[time]:.0f}_{fields}.png"

    fields_keys = list(fields)
    fields_names = []
    for key in fields_keys:
        fields_names.append(qg.FIELDS[key]) 
    qg.draw_snapshot(ds, time, path=snap_path, colorbar=True, fields=fields_names, show=show, size=size)
    return ds

if __name__ == '__main__':
    typer.run(main)