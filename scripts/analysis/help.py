import typer
import qgtoolbox as qg

def main(n_epoch: int = None, experiment: str = 'atm', short: bool = False):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    if n_epoch is None:
        data_path = path / f"{experiment}"
    else:
        data_path = path / f"{experiment}_{n_epoch}e"
    ds = qg.load_dataset(f"{data_path}.nc")
    ds = qg.calc_diganostic(ds)

    print(ds['q'])
    if short:
        ds.info()
    else:
        print(ds)
    return ds

if __name__ == '__main__':
    typer.run(main)