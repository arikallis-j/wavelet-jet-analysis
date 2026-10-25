import typer
import qgtoolbox as qg

def main(n_epoch: int = 1, exp: str = 'atm', save: str = 'e'):
    dm = qg.DataManager()
    sim_path = dm.make_dir(path=dm.raw, name=exp)
    res_path = dm.make_dir(path=dm.processed, name=exp)
    cfg_path = dm.config / f"{exp}.yaml"
    ds_path = sim_path / f"{exp}.nc"
    config = qg.parse_yaml(cfg_path)
    model = qg.make_bt_model(config)
    data = qg.run_simulation(model, n_epoch=n_epoch, path=ds_path, config=config, save_mode=save)
    return data

if __name__ == '__main__':
    typer.run(main)