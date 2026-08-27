using YAML
using Oceananigans


function parse_config(path::String)

    config = YAML.load_file(path)

    # Architecture
    architecture = config["architecture"]

    arch =
        if architecture == "CPU"
            CPU()
        elseif architecture == "GPU"
            GPU()
        else
            error("Unknown architecture: $architecture")
        end


    # Precision
    precision_name = config["grid"]["precision"]

    precision =
        if precision_name == "Float64"
            Float64
        elseif precision_name == "Float32"
            Float32
        else
            error("Unknown precision: $precision_name")
        end


    # Grid
    n_grid = Int(config["grid"]["n_grid"])


    # Advection
    advection_name = config["model"]["advection"]

    advection =
        if advection_name == "WENO"
            WENO()
        elseif advection_name == "Centered"
            Centered(order = 4)
        else
            error("Unknown advection scheme: $advection_name")
        end


    # Timestepper
    timestepper_name = config["model"]["timestepper"]

    timestepper =
        if timestepper_name == "RK3"
            :RungeKutta3
        else
            error("Unknown timestepper: $timestepper_name")
        end


    # Simulation
    Δt = Float64(config["simulation"]["dt"])
    stop_iteration = Int(config["simulation"]["stop_iteration"])


    return (
        arch = arch,
        precision = precision,
        n_grid = n_grid,
        advection = advection,
        timestepper = timestepper,
        Δt = Δt,
        stop_iteration = stop_iteration,
    )
end