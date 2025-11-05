using MPOPF
using Ipopt
using JuMP
using CSV
using DataFrames
using PowerModels
using IOCapture
using Random
const PM = PowerModels

include("generateD.jl")

scenarios = 2

function getData(foldertosave::String,folder::String, model_type::String)
    #folder = joinpath(folder, "matpower8.0/data")
    dc = (model_type == "DC") ? true : false
    for file in readdir(folder)
        case_number = nothing
        file_path = joinpath(folder, file)
        println("Processing file: $file_path")
        filename = splitext(file)[1]
        extension = splitext(file)[2]
        if (extension == ".m")
            filename = joinpath(foldertosave, filename)
            csvfilename = "$(filename).csv"
            csvfilenamePG = "$(filename)-pg.csv"

            m = match(r"case.*?(\d+)", file)   # look for "Case" followed by digits
            if m !== nothing
                case_number = parse(Int, m.captures[1])
            end

            if dc
                open(csvfilename, "w") do io
                    write(io, "\nStatus,Bus_from,Bus_to,volatge_magnitude_from,volatge_magnitude_to,theta_from,theta_to,cost,p_fr\n")
                end
            else
                open(csvfilename, "w") do io
                    write(io, "\nStatus,Bus_from,Bus_to,volatge_magnitude_from,volatge_magnitude_to,theta_from,theta_to,cost,p_fr,q_fr,p_to,q_to\n")
                end
            end

            open(csvfilenamePG, "w") do io
                write(io, "Status,Index,GeneratorBus,PowerGenerated,ReactivePowerGenerated\n")
            end

            generateDValues(case_number, model_type)

            open("Cases/$(scenarios)d_verify.csv", "w") do io
                write(io, "d\n")
            end

            if extension == ".m"
                #Random.seed!(1234)
                for j in 1:scenarios
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
                        status = "Feasible"
                    else
                        status = "Infeasible"
                        cost = "-"
                    end
                        #----------------------------------i Indexed Data ---------------------------------

                        #value for power generated
                        pg_val = JuMP.value.(My_AC_model.model[:pg])
                        qg_val = dc ? 0 : JuMP.value.(My_AC_model.model[:qg])

                        x = PowerModels.build_ref(data)[:it][:pm][:nw][0]
                        gen_data = x[:gen]

                        for i in pg_val.axes[2]
                            gen_bus = gen_data[i]["gen_bus"]
                            pg_at_i = pg_val[1, i]
                            qg_at_i =  dc ? 0 : qg_val[1, i]

                            open(csvfilenamePG, "a") do io
                                write(io, "$status,$i,$gen_bus,$pg_at_i,$qg_at_i\n")
                            end
                            
                        end

                        open(csvfilenamePG, "a") do io
                            write(io, "\n")
                        end

                        #---------------------------------------Branch data----------------------------------

                        #value for voltage amplitude
                        va_val = JuMP.value.(My_AC_model.model[:va])
                        vm_val =  dc ? Dict() : JuMP.value.(My_AC_model.model[:vm])

                        for (i, branch) in x[:branch]
                            f_bus = branch["f_bus"]
                            vm_from =  dc ? 0 : vm_val[1, f_bus]
                            va_from = va_val[1, f_bus]

                            t_bus = branch["t_bus"]
                            vm_to =  dc ? 0 : vm_val[1, t_bus]
                            va_to = va_val[1, t_bus]


                            f_idx = (i, branch["f_bus"], branch["t_bus"])
                            t_idx = dc ? (0,0,0) : (i, branch["t_bus"], branch["f_bus"])

                            p_fr = value(powerfrom[f_idx])
                            q_fr = dc ? "-" : value(reactancefrom[f_idx])

                            p_to = dc ? "-" : value(powerto[t_idx])
                            q_to = dc ? "-" : value(reactanceto[t_idx])

                            if dc == true
                                open(csvfilename, "a") do io
                                    write(io, "$status,$f_bus,$t_bus,$vm_from,$vm_to,$va_from,$va_to,$cost,$p_fr\n")
                                end
                            else
                                open(csvfilename, "a") do io
                                    write(io, "$status,$f_bus,$t_bus,$vm_from,$vm_to,$va_from,$va_to,$cost,$p_fr,$q_fr,$p_to,$q_to\n")
                                end
                            end
                        end

                    open(csvfilename, "a") do io
                        write(io, "\n")
                    end
                end
                if compareD() 
                    println("Same")
                else
                    error("Did not read correctly, from d values file $file")              
                end
            end
        end
    end
end

function generateDValues(case_number::Int, model_type::String)
    #copy from the generated file to $(scenarios)d.csv to be used bu all other method
    #cp("Cases/test/data/dvalues/Cases$(case_number)d.csv", "Cases/$(scenarios)d.csv"; force = true)
    if(model_type == "AC")
        println("Generating d values")
        genDValues(case_number, scenarios)
    end

    src = joinpath("Cases", "test", "data", "dvalues", "Cases$(case_number)d.csv")
    dest = joinpath("Cases", "$(scenarios)d.csv")

    cp(src, dest; force=true)
end

function runModel(model_type::String, file_path::String, j::Int64)
    My_model = nothing
    factory = nothing
    data = PowerModels.parse_file(file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    if(model_type == "AC")
        factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)
    elseif(model_type == "Approx")
        factory = LinTMPOPFModelFactory(file_path, Ipopt.Optimizer)
    else
        factory = DCMPOPFModelFactory(file_path, Ipopt.Optimizer)
    end
    
    My_model = create_model_demand(factory; i = j)

    return My_model, data
end

function compareD()
    #comparing d values to check if it was used correctly
    df1 = CSV.read("Cases/$(scenarios)d_verify.csv", DataFrame)
    df_unique = unique(df1)
    CSV.write("Cases/$(scenarios)d_verify.csv", df_unique) 
    
    df2 = CSV.read("Cases/$(scenarios)d.csv", DataFrame)
    df2_unique = unique(df2)
    return (isequal(df_unique, df2_unique) ? true : false)
end

function runGen()
    folder = "Cases/test"
    
    foldertosave = joinpath(folder, "data/AC")
    mkpath(foldertosave)
    getData(foldertosave, folder, "AC")
    
    run(`powerenv/bin/python3 src/getCoefficients.py`)
    #run(`powerenv/bin/python3 src/updateCoefficients.py`)

    foldertosave = joinpath(folder, "data/Approx")
    mkpath(foldertosave)
    getData(foldertosave, folder, "Approx")
    
    foldertosave = joinpath(folder, "data/DC")
    mkpath(foldertosave)
    getData(foldertosave, folder, "DC")
    

end

runGen()