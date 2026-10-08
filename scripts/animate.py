import typer
import qgtoolbox as qg

def main(n_epoch: int = None, experiment: str = 'atm', fps: int = 10, show: bool = False, fields: str ='qv', progress: bool = False,  size: int = 8,  j: int = 1, wtype: str = 'haar'):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    if n_epoch is None:
        data_path = path / f"{experiment}"
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    ds = qg.load_dataset(f"{data_path}.nc")
    ds = qg.calc_diganostic(ds, wtype=wtype)
    anim_path = f"{data_path}_{fields}.mp4"

    fields_keys = list(fields)
    fields_names = []
    kwargs = {
        'ew': {
            'j': j,
        }
    }
    for key in fields_keys:
        fields_names.append(qg.FIELDS[key]) 
    qg.draw_animation(ds, path=anim_path, fps=fps, colorbar=True, fields=fields_names, show=show, progress=progress, size=size, kwargs=kwargs)
    return ds

if __name__ == '__main__':
    typer.run(main)