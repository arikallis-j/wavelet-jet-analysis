using YAML, Oceananigans

function parse_config(config::String)
    config = YAML.load_file(config)
    
    arch_str = config["architecture"]
    arch = arch_str == "CPU" ? CPU() :
           arch_str == "GPU" ? GPU() :
           error("Unknown architecture: $arch_str")
    
    prec_str = config["grid"]["precision"]
    precision = prec_str == "Float64" ? Float64 :
                prec_str == "Float32" ? Float32 :
                error("Unknown precision: $prec_str")
    
    n_grid = config["grid"]["n_grid"]
    
    adv_str = config["model"]["advection"]
    advection = if adv_str == "WENO"
        WENO()
    elseif adv_str == "Centered"
        Centered(order=4)  # можно вынести в конфиг
    else
        error("Unknown advection scheme: $adv_str")
    end
    
    ts_str = config["model"]["timestepper"]
    timestepper = if ts_str == "RK3"
        :RungeKutta3
    elseif ts_str == "RK4"
        :RungeKutta4
    else
        error("Unknown timestepper: $ts_str")
    end
    
    Δt = config["simulation"]["dt"]
    stop_iteration = config["simulation"]["stop_iteration"]
    
    return (arch=arch, precision=precision, n_grid=n_grid,
            advection=advection, timestepper=timestepper,
            Δt=Δt, stop_iteration=stop_iteration)
end
