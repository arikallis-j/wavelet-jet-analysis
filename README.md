# wavelet-jet-analysis

> Wavelet analysis of zonal jets dynamics

This repository contains Julia (`Oceananigans`) simulations and Python post-processing, including Fourier and wavelet analysis of zonal jet dynamics.

## Basic scripts

run simulation:
```bash
python -m scripts.run --n-epoch=1 --experiment=iso-atm --save-mode=a
```

check dataset:
```bash
python -m scripts.help --n-epoch=1 --experiment=iso-atm --short
```

check one snapshot:
```bash
python -m scripts.check --n-epoch=1 --experiment=iso-atm --time=-1 --fields=pqv --show
```

make animation:
```bash
python -m scripts.animate --n-epoch=1 --experiment=iso-atm --fields=pqv --progress
```

## Setup enviroment

Create conda enviroment:

```bash
conda env create -f environment/conda_env.yaml
```

Update conda enviroment:

```bash
conda env update -f environment/conda_env.yaml
```

Download pip dependencies:

```bash
cd environment && pip install -r requirements.txt && cd ..
```

Freeze current python dependencies:

```bash
pip list --not-required --format=freeze > environment/backup/requirements.txt.backup
```

Download julia dependencies:

```bash
julia -e 'using Pkg; packages = filter(!isempty, readlines("environment/julia_packages.txt")); Pkg.add(packages)'
```

Freeze current julia dependencies:

```bash
PROJECT=$(julia -e 'print(Base.active_project())') && cp "$PROJECT" environment/backup/Project.toml && cp "${PROJECT/../Manifest.toml}" environment/backup/Manifest.toml
```

## How to run workflow

```bash
snakemake <rule> --quiet all -j 1 --config <param>=<value>
```

where `<rule>`:

- `all` or ` `
- `run_simulation`
- `fit_parameters`
- `plot_figure`

or by specific pipeline:

```bash
snakemake -s workflow/<name>.smk --quiet all -j 1 --config <param>=<value>
```
