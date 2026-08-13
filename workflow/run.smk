# workflow/run.smk
        
rule run_simulation:
    shell:
        "julia jets/simulation/run_simulation.jl"