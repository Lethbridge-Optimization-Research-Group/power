using MPOPF
using Ipopt
using JuMP
using CSV
using DataFrames
using PowerModels
using IOCapture
using Random
const PM = PowerModels

folder = "Cases/test"
foldertosave = joinpath(folder, "data")

for file in readdir(folder)
    file_path = joinpath(folder, file)
    println("Processing file: $file_path")
    filename = splitext(file)[1]
    extension = splitext(file)[2]
    if (extension == ".m")
        filename = joinpath(foldertosave, filename)
        csvfilename = "$(filename).csv"
        csvfilenamePG = "$(filename)-pg.csv"
        csvfilenamelin = "$(filename)-linear.csv"
        csvfilenamelinPG = "$(filename)-linear-pg.csv"
        open(csvfilename, "w") do io
            write(io, "\nBus_from,Bus_to,volatge_magnitude_from,volatge_magnitude_to,theta_from,theta_to,cost \n")
        end
        
        open(csvfilenamePG, "w") do io
            write(io, "Index,GeneratorBus,PowerGenerated,ReactivePowerGenerated\n")
        end

        open(csvfilenamelin, "w") do io
            write(io, "\nBus_from,Bus_to,volatge_magnitude_from,volatge_magnitude_to,theta_from,theta_to,cost \n")
        end
        
        open(csvfilenamelinPG, "w") do io
            write(io, "Index,GeneratorBus,PowerGenerated,ReactivePowerGenerated\n")
        end

        if extension == ".m"
            Random.seed!(1234)
            for i in 1:10
                My_AC_model = nothing
                data = nothing
                cost = nothing

                #optimize_model(My_AC_model)
                output = IOCapture.capture() do
                    data = PowerModels.parse_file(file_path)
                    PowerModels.standardize_cost_terms!(data, order=2)
                    PowerModels.calc_thermal_limits!(data)

                    ac_factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)
                    My_AC_model = create_model(ac_factory)
                    optimize_model(My_AC_model)
                end

                lines = split(output.output, '\n')
                #for (i, line) in enumerate(lines)
                #    println("[$i] --> ", repr(line))
                #end

                exit_line = findfirst(startswith("EXIT:"), lines)
                if exit_line !== nothing && lines[exit_line] == "EXIT: Optimal Solution Found."
                    println("Found EXIT line: ", lines[exit_line])               
                    
                    cost_line_index = exit_line +1
                    if cost_line_index !== nothing
                        cost_line = lines[cost_line_index]
                        m = match(r"Optimal Cost:\s+([0-9.]+)", cost_line)
                        if m !== nothing
                            cost = parse(Float64, m.captures[1])
                            #println("Extracted cost: ", cost)
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
                    #println("Infeasible")
                    open(csvfilename, "a") do io
                        write(io, "Infeasible\n")
                    end
                end
            end
            
            Random.seed!(1234)
            for i in 1:10
                My_ACT_model = nothing
                data = nothing
                cost = nothing

                #optimize_model(My_AC_model)
                output2 = IOCapture.capture() do
                    data = PowerModels.parse_file(file_path)
                    PowerModels.standardize_cost_terms!(data, order=2)
                    PowerModels.calc_thermal_limits!(data)

                    act_factory = LinTMPOPFModelFactory(file_path, Ipopt.Optimizer)
                    My_ACT_model = create_model(act_factory)
                    optimize_model(My_ACT_model)
                end

                lines2 = split(output2.output, '\n')
                #for (i, line) in enumerate(lines)
                #    println("[$i] --> ", repr(line))
                #end

                exit_line = findfirst(startswith("EXIT:"), lines2)
                if exit_line !== nothing && lines2[exit_line] == "EXIT: Optimal Solution Found."
                    println("Found EXIT line: ", lines2[exit_line])               
                    
                    cost_line_index = exit_line +1
                    if cost_line_index !== nothing
                        cost_line = lines2[cost_line_index]
                        m = match(r"Optimal Cost:\s+([0-9.]+)", cost_line)
                        if m !== nothing
                            cost = parse(Float64, m.captures[1])
                            #println("Extracted cost: ", cost)
                        else
                            println("Cost format not matched.", cost_line)
                        end
                    else
                        println("Cost line not found")
                    end


                    #----------------------------------i Indexed Data ---------------------------------

                    #value for power generated
                    pg_val = JuMP.value.(My_ACT_model.model[:pg])
                    qg_val = JuMP.value.(My_ACT_model.model[:qg])

                    x = PowerModels.build_ref(data)[:it][:pm][:nw][0]
                    gen_data = x[:gen]

                    for i in pg_val.axes[2]
                        gen_bus = gen_data[i]["gen_bus"]
                        pg_at_i = pg_val[1, i]
                        qg_at_i = qg_val[1, i]

                        open(csvfilenamelinPG, "a") do io
                            write(io, "$i,$gen_bus,$pg_at_i,$qg_at_i\n")
                        end
                        
                    end

                    open(csvfilenamelinPG, "a") do io
                        write(io, "\n")
                    end

                    #---------------------------------------Branch data----------------------------------

                    #value for voltage amplitude
                    va_val = JuMP.value.(My_ACT_model.model[:va])
                    vm_val = JuMP.value.(My_ACT_model.model[:vm])

                    for (i, branch) in x[:branch]
                        f_bus = branch["f_bus"]
                        vm_from = vm_val[1, f_bus]
                        va_from = va_val[1, f_bus]

                        t_bus = branch["t_bus"]
                        vm_to = vm_val[1, t_bus]
                        va_to = va_val[1, t_bus]

                        open(csvfilenamelin, "a") do io
                            write(io, "$f_bus,$t_bus,$vm_from,$vm_to,$va_from,$va_to,$cost\n")
                        end 
                    end
                else
                    #println("Infeasible")
                    open(csvfilenamelin, "a") do io
                        write(io, "Infeasible\n")
                    end
                end
            end
        end
        
    end
end