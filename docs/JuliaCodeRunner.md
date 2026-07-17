# julia-code-runner

## access to terminal arguments

```bash
julia run.jl 1000 0.1
```

```julia
println(ARGS)
> ["1000", "0.1"]

N = parse(Int, ARGS[1])
dt = parse(Float64, ARGS[2])
println("N = $N, dt = $dt")
> N = 1000, dt = 0.1
```

example of pipeline:

```bash
for k in 20 40 80
do
    julia --project=. run.jl $k
done
```

## code run in number of threads

```bash
$ julia --project=. --threads=8 run.jl
```

## include code in REPL

simple include of code (don't see changes in file)

```bash
$ julia
julia> include("script.jl")
julia> x = rand(10)
...
# some changes in `script.jl`
julia> include("script.jl")
julia> x = rand(10)
```

Julia package `Revise` (see changes in file)

```bash
$ julia
julia> using Revise
julia> include("script.jl")
julia> x = rand(10)
...
# some changes in `script.jl`
julia> x = rand(10)
```

## jupyter notebook

Jupyter Notebook works with package `IJulia`

```bash
$ julia
julia> using IJulia
julia> notebook()
```

## pluto notebook

Pluto Notebook works with package `Pluto`

```bash
$ julia
julia> using Pluto
julia> Pluto.run()
```

## tmux runner

### sessions

```bash
# start new session `ocean`
tmux new -s ocean

# list of all sessions
tmux ls

# connect to session `ocean`
tmux attach -t

# disconnect from session
tmux detach # `Ctrl + b` & `d`

# rename session `ocean` to `new-ocean`
tmux rename-session -t ocean new-ocean # `Ctrl + b` & `$`

# delete session `ocean`
tmux kill-session -t ocean
```

### windows

`Ctrl + b` & `d` - disconnect from session
`Ctrl + b` & `$` - rename session

`Ctrl + b` & `c` - create new window
`Ctrl + b` & `&` - close the current window
`Ctrl + b` & `,` - rename the current window

`Ctrl + b` & `n` - switch to the next window
`Ctrl + b` & `n` - switch to the previuos window
`Ctrl + b` & `0-9` - switch to the num window

### panes

`Ctrl + b` & `"` - divide the window horizontally
`Ctrl + b` & `%` - divide the window vertically
`Ctrl + b` & `x` - close the current pane
`Ctrl + b` & `↑` `↓` `←` `→` - switch between panes

### others

`Ctrl + b` & `[` - scrollback in output history
`Ctrl + b` & `t` - show the clocks (hours:minutes)
`Ctrl + b` & `?` - show list of hotkeys
`q` - quit from tmux regime

## Slurm runner

> Slurm (Simple Linux Utility for Resource Management)

The next step after tmux, for claster calculations

example:

```bash
sbatch run.sh
```

```shell
#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

julia --project=. run.jl
```

## Julia Compiler

Package`JuliaC` create binary file for julia code `.jl`

```bash
pkg> app add JuliaC
```

```julia
# my_script.jl
function @main(args::Vector{String})
    println("Hello from compiled Julia!")
    println("Arguments: ", args)
    # ... ваш код здесь
end
```

```bash
juliac --output-exe my_program --bundle build --trim=safe --experimental my_script.jl
```
