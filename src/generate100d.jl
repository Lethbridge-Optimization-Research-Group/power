using CSV
using Random
using Distributions

function genDValues()
    csvfilename = "Cases/100d.csv"
    d = nothing
    open(csvfilename, "w") do io
        write(io, "d\n")
    end

    for i in 1:100
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
        sigma = 0.1
        d = rand(Normal(mu, sigma))
    end
    return d
end

genDValues()