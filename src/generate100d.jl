using CSV
using Random
using Distributions

function genDValues(val::Int)
    csvfilename = "Cases/test/data/dvalues/Cases$(val)d.csv"
    d = nothing
    open(csvfilename, "w") do io
        write(io, "d\n")
    end
    val = val * 100
    for i in 1:val
        d = sampling("Normal") 
        open(csvfilename, "a") do io
            write(io, "$d\n")
        end
    end
end

function sampling(i::String)
    d = nothing
    if i == "Uniform"
        d = rand(Uniform(.9,1.1))
    else
        mu = 1
        sigma = 0.05
        d = rand(Normal(mu, sigma))
    end
    return d
end

#genDValues(5)