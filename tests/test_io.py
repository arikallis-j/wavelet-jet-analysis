import qgtoolbox as qg
import typer

def main(n_epoch: int = None, exp: str = 'iso-atm'):
    dm = qg.DataManager()
    print(dm.root)
    print(dm.data)
    print(dm.config)
    print(dm.raw)
    print(dm.processed)

    sim_path = dm.make_dir(path=dm.raw, name=exp)
    res_path = dm.make_dir(path=dm.processed, name=exp)
    stat_path = dm.make_dir(path=res_path, name='stat')

    cfg_path = dm.config / f"{exp}.yaml"
    if n_epoch is None:
        ds_path = sim_path / f"{exp}.nc"
    else:
        ds_path = sim_path / f"{exp}_{n_epoch}e.nc"

    print(sim_path)
    print(res_path)
    print(stat_path)
    print(cfg_path)
    print(ds_path)
    
    config = qg.parse_yaml(path=cfg_path)
    print(config)

    ds = qg.load_dataset(path=ds_path)
    print(ds)

    ds_new = ds.isel(time=-1)
    print(ds)
    qg.save_dataset(data=ds_new, path=stat_path / f"last.nc")

    return None

if __name__ == '__main__':
    typer.run(main)