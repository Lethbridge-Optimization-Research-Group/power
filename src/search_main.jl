using JuMP, Ipopt, Gurobi, Serialization, Random, Graphs, MetaGraphs, MathOptInterface
using PowerModels
using MPOPF
using Statistics, Plots, GraphRecipes

include("graph_search.jl")
include("rampingCSVimplementation.jl")


#matpower_file_path = "./Cases/case14.m" 
#matpower_file_path = "./Cases/case300.m" 
matpower_file_path = "./Cases/case9241pegase.m" 
#matpower_file_path = "./Cases/case1197.m" 
#matpower_file_path = "./Cases/case_ACTIVSg200.m" 
#matpower_file_path = "./Cases/case_ACTIVSg500.m" 
#matpower_file_path = "./Cases/case_ACTIVSg2000.m"
#matpower_file_path = "./Cases/pglib_opf_case500_goc.m"
#matpower_file_path = "./Cases/pglib_opf_case793_goc.m" - does not run
#matpower_file_path = "./Cases/Wisconsin_1664.m" - no improvement for both models
#matpower_file_path = "./Cases/pglib_opf_case1803_snem.m"
#matpower_file_path = "./Cases/pglib_opf_case1888_rte.m"
#matpower_file_path = "./Cases/pglib_opf_case1951_rte.m"
#matpower_file_path = "./Cases/pglib_opf_case2000_goc.m" - does not run
#matpower_file_path = "./Cases/pglib_opf_case2312_goc.m"
#matpower_file_path = "./Cases/pglib_opf_case2383wp_k.m"

t = 3

output_dir = "./Cases"
data = PowerModels.parse_file(matpower_file_path)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)
#=
for (gen_id, gen_data) in data["gen"]
    if gen_data["pmin"] < 0
        gen_data["pmin"] = 0
    end
end
=#
ramping_csv_file = generate_daily_demand_csv(data, output_dir)
ramping_data, demands = parse_power_system_csv(ramping_csv_file, matpower_file_path)

search_factory = DCMPOPFSearchFactory(matpower_file_path, Gurobi.Optimizer)
search_model = create_search_model(search_factory, t, ramping_data, demands)
opt_start = time()
optimize!(search_model.model)
opt_stop = time()
println(opt_stop - opt_start, " seconds")

search_start = time()
info = DC_graph_search(data, search_factory, demands, ramping_data, t)
search_stop = time()
println()

opt = objective_value(search_model.model)
diff = info[:cost] / opt
 
println("Seconds: ", search_stop - search_start)
println("Difference: ", info[:cost] / opt) 

filename = split(matpower_file_path, "/") |> last

graph_demands_and_generation(demands, search_model, info[:solution])
output_run_data_to_csv(data, matpower_file_path, demands, search_model, info)

 # If wanting to graph the Cost History
opt = objective_value(search_model.model)

plot(info[:cost_history], 
     label="Optimization Cost", 
     title="Cost History : $matpower_file_path",
     xlabel="Iteration", 
     ylabel="Cost",
     linewidth=2,
     marker=:circle,
     markersize=3)
annotate!(1, maximum(info[:cost_history]), text("Opt: = $opt", :left, 10))
title = time()
#savefig("second_iteratoin.png")

# If wanting to test with PowerModelds PF
#test_model = PowerModels.solve_pf(data, DCPPowerModel, Gurobi.Optimizer)


models = []
for i in 1:t
     model = create_search_model(search_factory, 1, ramping_data, [demands[i]])
     optimize!(model.model)
     push!(models, model)
end

pg_values_by_t = []
for i in 1:t
    values = [value(models[i].model[:pg][key]) for key in keys(models[i].model[:pg])]
    pg_values = Dict(zip(models[i].model[:pg].axes[2], values))
    push!(pg_values_by_t, pg_values)
end

# Calculate total ramping cost
global total_ramping_cost = 0.0
for t in 2:t
    for gen_id in ramping_data["gen_id"]
        gen_key = Int(gen_id)
        current_power = pg_values_by_t[t][gen_key]
        previous_power = pg_values_by_t[t-1][gen_key]
        ramp_amount = abs(current_power - previous_power)
        global total_ramping_cost += ramp_amount * ramping_data["costs"][gen_key]
    end
end

println("Total ramping cost: $total_ramping_cost")

#=
for t in 1:5
    x = info[:solution][t]
    println()
    for (gen_id, gen_data) in data["gen"]
        gen_id_int = parse(Int, gen_id)
        if x[:generator_values][gen_id_int] < gen_data["pmin"]
            println("Pmin violation on $gen_id")
        end
        if x[:generator_values][gen_id_int] > gen_data["pmax"]
            println("Pmax violation on $gen_id")
        end 
    end 
end

for t in 1:5
    x = info[:solution][t]
    println()
    for (gen_id, gen_data) in data["gen"]
        gen_id = parse(Int, gen_id)
        println("Gen ID: $gen_id")
        println("Gen pmin: ", gen_data["pmin"])
        println("Gen Pg: ", x[:generator_values][gen_id])
        println()
    end 
end
=#