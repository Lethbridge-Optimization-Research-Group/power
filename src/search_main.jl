using JuMP, Ipopt, Gurobi, Serialization, Random, Graphs, MetaGraphs, MathOptInterface
using PowerModels
using MPOPF
using Statistics, Plots, GraphRecipes

include("graph_search.jl")
include("rampingCSVimplementation.jl")

matpower_file_path = "./Cases/case14.m"

output_dir = "./Cases"
data = PowerModels.parse_file(matpower_file_path)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

ramping_csv_file = generate_daily_demand_csv(data, output_dir)
ramping_data, demands = parse_power_system_csv(ramping_csv_file, matpower_file_path)

search_factory = DCMPOPFSearchFactory(matpower_file_path, Gurobi.Optimizer)
search_model = create_search_model(search_factory, 24, ramping_data, demands)
opt_start = time()
optimize!(search_model.model)
opt_stop = time()
println(opt_stop - opt_start, " seconds")

search_start = time()
info = DC_graph_search(data, search_factory, demands, ramping_data, 24)
search_stop = time()
println()

opt = objective_value(search_model.model)
diff = info[:cost] / opt
 
println("Seconds: ", search_stop - search_start)
println("Difference: ", info[:cost] / opt) 

filename = split(matpower_file_path, "/") |> last

graph_demands_and_generation(demands, search_model, info[:solution])
output_run_data_to_csv(data, matpower_file_path, demands, search_model, info)

#= If wanting to graph the Cost History
opt = objective_value(search_model.model)

plot(cost_history, 
     label="Optimization Cost", 
     title="Cost History During Optimization : $matpower_file_path",
     xlabel="Iteration", 
     ylabel="Cost",
     linewidth=2,
     marker=:circle,
     markersize=3)
annotate!(1, maximum(cost_history), text("Opt: = $opt", :left, 10))
title = time()
savefig("$title.png")
=#

# If wanting to test with PowerModelds PF
#test_model = PowerModels.solve_pf(data, DCPPowerModel, Gurobi.Optimizer)

#TODO: Test feasibility for largest demand time period over all time periods - Done
#TODO: Iterate through time periods and test feasibility
#TODO: Test largest time period model for constraints - Done (Constraints not violated)
#TODO: Give largest time period models to graph model fixed - Done

#TODO: Run OPF on individual time periods and calculate ramping costs afterwards
#      Remove ramping costs in the objective function
#      Compare these ramping costs to the graph model ramping costs

#TODO: If finding the largest time period and putting it over all time periods
#      doesn't work, maybe try giving it a hot start some % above the actual demand

#TODO: Select and change a subset of generators