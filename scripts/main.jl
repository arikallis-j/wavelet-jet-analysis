println(ARGS)

N = parse(Int, ARGS[1])
dt = parse(Float64, ARGS[2])
println("N = $N, dt = $dt")

N = 10
for k in 1:10
    println(k)
end