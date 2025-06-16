using MPOPF
using Ipopt
using JuMP
using CSV
using DataFrames
using PowerModels
const PM = PowerModels

folder = "/home/prottasha-dcruze/UniversityProjects/OpticalPowerFlow/power/Cases/test"

for file in readdir(folder)
    file_path = joinpath(folder, file)
    println("Processing file: $file_path")
    filename = splitext(file)[1]
    extension = splitext(file)[2]
    filename = joinpath(folder, filename)
    csvfilename = "$(filename).csv"

    if extension == ".m"
        data = PowerModels.parse_file(file_path)
        PowerModels.standardize_cost_terms!(data, order=2)
        PowerModels.calc_thermal_limits!(data)

        ac_factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)
        My_AC_model = create_model(ac_factory)
        #optimize_model(My_AC_model)

        optimize!(My_AC_model.model)

        #----------------------------------i Indexed Data ---------------------------------

        #value for power generated
        pg_val = JuMP.value.(My_AC_model.model[:pg])
        qg_val = JuMP.value.(My_AC_model.model[:qg])

        x = PowerModels.build_ref(data)[:it][:pm][:nw][0]
        gen_data = x[:gen]

        open(csvfilename, "w") do io
            write(io, "Index,GeneratorBus,PowerGenerated,ReactivePowerGenerated\n")
        end

        for i in pg_val.axes[2]
            gen_bus = gen_data[i]["gen_bus"]
            pg_at_i = pg_val[1, i]
            qg_at_i = qg_val[1, i]

            open(csvfilename, "a") do io
                write(io, "$i,$gen_bus,$pg_at_i,$qg_at_i\n")
            end
            
        end

        open(csvfilename, "a") do io
            write(io, "\n")
        end

        #----------------------------------Bus Indexed Data ---------------------------------

        #value for voltage amplitude
        va_val = JuMP.value.(My_AC_model.model[:va])
        vm_val = JuMP.value.(My_AC_model.model[:vm])


        open(csvfilename, "a") do io
            write(io, "Bus,VoltageAngle,VolatgeMagnitude\n")
        end

        for bus in va_val.axes[2]
            va_at_bus= va_val[1, bus]
            vm_at_bus= vm_val[1, bus]

            open(csvfilename, "a") do io
                write(io, "$bus,$va_at_bus,$vm_at_bus\n")
            end
        end

        #---------------------------------------Branch data----------------------------------
        
        open(csvfilename, "a") do io
            write(io, "\nBus_from,Bus_to,volatge_magnitude_from,volatge_magnitude_to,theta_from,theta_to,cos_theta,sin_theta,vm_cos_theta,vm_sin_theta\n")
        end

        for (i, branch) in x[:branch]
            f_bus = branch["f_bus"]
            vm_from = vm_val[1, f_bus]
            va_from = va_val[1, f_bus]

            t_bus = branch["t_bus"]
            vm_to = vm_val[1, t_bus]
            va_to = va_val[1, t_bus]

            cos_theta = cos(va_from - va_to)
            sin_theta = sin(va_from - va_to)

            vm_cos_theta = vm_from*vm_to*cos_theta
            vm_sin_theta = vm_from*vm_to*sin_theta

            open(csvfilename, "a") do io
                write(io, "$f_bus,$t_bus,$vm_from,$vm_to,$va_from,$va_to,$cos_theta,$sin_theta,$vm_cos_theta,$vm_sin_theta\n")
            end 
        end
    end
end