using MPOPF
using Ipopt
using JuMP
using CSV
using DataFrames
using PowerModels
using IOCapture
using Random
const PM = PowerModels

function getData(foldertosave::String,folder::String, model_type::String)
    for file in readdir(folder)
        file_path = joinpath(folder, file)
        println("Processing file: $file_path")
        filename = splitext(file)[1]
        extension = splitext(file)[2]
        if (extension == ".m")
            filename = joinpath(foldertosave, filename)
            csvfilename = "$(filename).csv"
            csvfilenamePG = "$(filename)-pg.csv"
            open(csvfilename, "w") do io
                write(io, "\nBus_from,Bus_to,volatge_magnitude_from,volatge_magnitude_to,theta_from,theta_to,cost \n")
            end
            
            open(csvfilenamePG, "w") do io
                write(io, "Index,GeneratorBus,PowerGenerated,ReactivePowerGenerated\n")
            end

            open("Cases/100d_verify.csv", "w") do io
                write(io, "d\n")
            end

            if extension == ".m"
                Random.seed!(1234)
                for j in 1:100
                    My_AC_model = nothing
                    data = nothing
                    cost = nothing

                    output = IOCapture.capture() do
                        My_AC_model, data = runModel(model_type, file_path, j)
                        optimize_model(My_AC_model)
                    end

                    lines = split(output.output, '\n')

                    exit_line = findfirst(startswith("EXIT:"), lines)
                    if exit_line !== nothing && lines[exit_line] == "EXIT: Optimal Solution Found."
                        println("Found EXIT line: ", lines[exit_line])               
                        
                        cost_line_index = exit_line +1
                        if cost_line_index !== nothing
                            cost_line = lines[cost_line_index]
                            m = match(r"Optimal Cost:\s+([0-9.]+)", cost_line)
                            if m !== nothing
                                cost = parse(Float64, m.captures[1])
                            else
                                println("Cost format not matched.", cost_line)
                            end
                        else
                            println("Cost line not found")
                        end

                        #----------------------------------i Indexed Data ---------------------------------

                        #value for power generated
                        pg_val = JuMP.value.(My_AC_model.model[:pg])
                        qg_val = JuMP.value.(My_AC_model.model[:qg])

                        x = PowerModels.build_ref(data)[:it][:pm][:nw][0]
                        gen_data = x[:gen]

                        for i in pg_val.axes[2]
                            gen_bus = gen_data[i]["gen_bus"]
                            pg_at_i = pg_val[1, i]
                            qg_at_i = qg_val[1, i]

                            open(csvfilenamePG, "a") do io
                                write(io, "$i,$gen_bus,$pg_at_i,$qg_at_i\n")
                            end
                            
                        end

                        open(csvfilenamePG, "a") do io
                            write(io, "\n")
                        end

                        #---------------------------------------Branch data----------------------------------

                        #value for voltage amplitude
                        va_val = JuMP.value.(My_AC_model.model[:va])
                        vm_val = JuMP.value.(My_AC_model.model[:vm])

                        for (i, branch) in x[:branch]
                            f_bus = branch["f_bus"]
                            vm_from = vm_val[1, f_bus]
                            va_from = va_val[1, f_bus]

                            t_bus = branch["t_bus"]
                            vm_to = vm_val[1, t_bus]
                            va_to = va_val[1, t_bus]

                            open(csvfilename, "a") do io
                                write(io, "$f_bus,$t_bus,$vm_from,$vm_to,$va_from,$va_to,$cost\n")
                            end 
                        end
                    else
                        open(csvfilename, "a") do io
                            write(io, "Infeasible\n")
                        end
                    end
                    open(csvfilename, "a") do io
                        write(io, "\n")
                    end
                end               
            end
        end
    end
end

function runModel(model_type::String, file_path::String, j::Int64)
    My_AC_model = nothing
    ac_factory = nothing
    data = PowerModels.parse_file(file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    if(model_type == "AC")
        ac_factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)
    else
        ac_factory = LinTMPOPFModelFactory(file_path, Ipopt.Optimizer)
    end

    My_AC_model = create_model_demand(ac_factory; i = j)

    return My_AC_model, data
end

function compareD()
    #comparing d values to check if it was used correctly
    df1 = CSV.read("Cases/100d_verify.csv", DataFrame)
    df_unique = unique(df1)
    CSV.write("Cases/100d_verify.csv", df_unique) 
    
    df2 = CSV.read("Cases/100d.csv", DataFrame)
    return (isequal(df_unique, df2) ? true : false)
end

function runGen()
    folder = "Cases/test"
    foldertosave = joinpath(folder, "data/AC")
    mkpath(foldertosave)

    getData(foldertosave, folder, "AC")
    compareD() ? println("Same") : println("Different")
    run(`powerenv/bin/python3 src/getCoefficients.py`)

    if compareD()
        foldertosave = joinpath(folder, "data/Approx")
        mkpath(foldertosave)
        getData(foldertosave, folder, "Approx")
        if compareD()
            println("All ran with same d")
        else
            println("AC ran with correct d, Approx did not")
        end
    else
        println("AC did not run with correct d")
    end
end

runGen()