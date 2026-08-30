import typer
import qgtoolbox as qg
import numpy as np

def main(n_epoch: int = None, experiment: str = 'atm', fps: int = 10, show: bool = False, fields: str ='qv', progress: bool = False):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    if n_epoch is None:
        data_path = path / f"{experiment}"
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    ds = qg.load_dataset(f"{data_path}.nc")
    ds['v'] = np.sqrt(ds['ux']**2 + ds['uy']**2)
    anim_path = f"{data_path}.mp4"
    qg.draw_animation(ds, path=anim_path, fps=fps, colorbar=True, fields=list(fields), show=show, progress=progress)
    return ds

if __name__ == '__main__':
    typer.run(main)