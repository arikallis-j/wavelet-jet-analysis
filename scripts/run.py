import typer
import qgtoolbox as qg

def main(n_epoch: int = 1, experiment: str = 'atm', save_mode: str = 'a'):
    dm = qg.DataManager()
    path = dm.make_experiment(experiment)
    config = qg.parse_yaml(dm.config/f"{experiment}.yaml")
    model = qg.make_bt_model(config)
    data = qg.run_simulation(model, n_epoch=n_epoch, path=path/f"{experiment}.nc", config=config, save_mode=save_mode)
    return data

if __name__ == '__main__':
    typer.run(main)