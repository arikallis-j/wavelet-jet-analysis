using Oceananigans
using Oceananigans.OutputWriters
using NCDatasets

include("../utils/parse_config.jl")


function main(config_path::String, output_path::String)

    # ------------------------------------------------------------
    # 1. Parse configuration
    # ------------------------------------------------------------

    @info "Parsing configuration: $config_path"

    params = parse_config(config_path)

    @info "Configuration parsed" params


    # ------------------------------------------------------------
    # 2. Initialize grid
    # ------------------------------------------------------------

    @info "Initializing grid"

    grid = RectilinearGrid(
        params.arch,
        params.precision;
        size = (params.n_grid, params.n_grid),
        x = (0, 2π),
        y = (0, 2π),
        topology = (Periodic, Periodic, Flat),
    )

    @info "Grid initialized" grid


    # ------------------------------------------------------------
    # 3. Initialize model
    # ------------------------------------------------------------

    @info "Initializing model"

    model = NonhydrostaticModel(
        grid;
        advection = params.advection,
        timestepper = params.timestepper,
    )

    @info "Model initialized"


    # ------------------------------------------------------------
    # 4. Initialize model state
    # ------------------------------------------------------------

    @info "Initializing model state"

    ϵ(x, y) = 2 * rand() - 1

    set!(
        model;
        u = ϵ,
        v = ϵ,
    )


    # ------------------------------------------------------------
    # 5. Define diagnostics / output fields
    # ------------------------------------------------------------

    @info "Initializing diagnostics"

    ζ = Field(
        ∂x(model.velocities.v) -
        ∂y(model.velocities.u)
    )

    outputs = Dict(
        "u" => model.velocities.u,
        "v" => model.velocities.v,
        "ζ" => ζ,
    )


    # ------------------------------------------------------------
    # 6. Initialize NetCDF writer
    # ------------------------------------------------------------

    @info "Initializing NetCDF writer"

    nc_writer = NetCDFWriter(
        model,
        outputs;
        filename = output_path,
        schedule = IterationInterval(100),
        overwrite_existing = true,
    )


    # ------------------------------------------------------------
    # 7. Initialize simulation
    # ------------------------------------------------------------

    @info "Initializing simulation"

    simulation = Simulation(
        model;
        Δt = params.Δt,
        stop_iteration = params.stop_iteration,
    )

    simulation.output_writers[:nc] = nc_writer


    # ------------------------------------------------------------
    # 8. Callbacks
    # ------------------------------------------------------------

    show_time(sim) =
        "Time is $(prettytime(sim.model.clock.time))"

    simulation.callbacks[:show_time] =
        Callback(show_time, IterationInterval(100))


    # ------------------------------------------------------------
    # 9. Run simulation
    # ------------------------------------------------------------

    @info "Starting simulation"

    run!(simulation)

    @info "Simulation finished"

    return simulation
end


# ----------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__

    if length(ARGS) != 2
        error(
            "Usage: julia run.jl <config_path> <output_path>"
        )
    end

    config_path = ARGS[1]
    output_path = ARGS[2]

    @time main(config_path, output_path)

end