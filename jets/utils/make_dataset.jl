using NCDatasets, Oceananigans

function make_dataset(grid, model, filename)
    u, v, w = model.velocities
    ζ = Field(∂x(v) - ∂y(u))
    compute!(ζ)

    xs = xnodes(grid, Center())
    ys = ynodes(grid, Center())

    u_data = interior(u)[:, :, 1]
    v_data = interior(v)[:, :, 1]
    ζ_data = interior(ζ)[:, :, 1]

    NCDataset(filename, "c") do ds
        defDim(ds, "x", length(xs))
        defDim(ds, "y", length(ys))
        defVar(ds, "x", xs, ("x",))
        defVar(ds, "y", ys, ("y",))
        defVar(ds, "u", u_data, ("x", "y"))
        defVar(ds, "v", v_data, ("x", "y"))
        defVar(ds, "ζ", ζ_data, ("x", "y"))
        defVar(ds, "time", model.clock.time, ())
    end
    println("Results saved to $filename")
end

