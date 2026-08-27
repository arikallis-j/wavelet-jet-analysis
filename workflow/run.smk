# workflow/run.smk
        
EXPERIMENT = config.get("experiment", "atm")
N_EPOCH = config.get("n-epoch", 1)
CONFIG_FILE = f"./config/{EXPERIMENT}.yaml"
DATA_FILE = f"./data/processed/{EXPERIMENT}.nc"
        
rule run_simulation:
    input:
        config = CONFIG_FILE
    output:
        data = DATA_FILE
    shell:
        "python scripts/run.py --n-epoch={N_EPOCH} --experiment={EXPERIMENT}"