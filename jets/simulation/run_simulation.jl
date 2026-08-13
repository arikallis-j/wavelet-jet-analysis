using Oceananigans, CairoMakie, NCDatasets

include("../utils/parse_config.jl")
include("../utils/make_dataset.jl")

if length(ARGS) < 1
    error("Usage: julia myscript.jl <config_name>")
end

config, filename = ARGS[1], ARGS[2]

params = parse_config(config)

grid = RectilinearGrid(
    params.arch,
    params.precision,
    size = (params.n_grid, params.n_grid),
    x = (0, 2π),
    y = (0, 2π),
    topology = (Periodic, Periodic, Flat)
)

model = NonhydrostaticModel(
    grid;
    advection = params.advection,
    timestepper = params.timestepper,
)

ϵ(x, y) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)

simulation = Simulation(
    model; 
    Δt = params.Δt, 
    stop_iteration = params.stop_iteration
)

run!(simulation)
make_dataset(grid, model, filename)
