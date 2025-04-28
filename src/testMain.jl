using JuMP, Ipopt, Gurobi, Serialization, Random, Graphs, MetaGraphs, MathOptInterface
using PowerModels
using MPOPF
using Statistics, Plots, GraphRecipes

include("graph_search.jl")
include("search_functions.jl")

matpower_file_path = "./Cases/case14.m"
#matpower_file_path = "./Cases/case300.m"
#matpower_file_path = "./Cases/case1354pegase.m"
#matpower_file_path = "./Cases/case9241pegase.m"
output_dir = "./Cases"
data = PowerModels.parse_file(matpower_file_path)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

#TODO: if stuck in local minima, make larger change / restart, etc
#TODO: Try and reuse models 
#TODO: Tweak search parameters, adjust variation
#TODO: See if we can use powermodels for PF 
#TODO: Look into running in parallel
#TODO: see if we can silence PowerModels output as well
#TODO: compare infeasible full model, see if feasibility improves w local search
#TODO: plot improvements between iterations (100-500 busses) - Done
#TODO: Run on some bigger cases, full model vs local search model with more iterations - Tried
#TODO: State of case9241, *try* same data on local search for a small number of iterations - Too large
ramping_csv_file = generate_power_system_csv(data, output_dir)
ramping_data, demands = parse_power_system_csv(ramping_csv_file, matpower_file_path)

search_factory = DCMPOPFSearchFactory(matpower_file_path, Gurobi.Optimizer)
search_model = create_search_model(search_factory, 5, ramping_data, demands)
opt_start = time()
optimize!(search_model.model)
opt_stop = time()
println(opt_stop - opt_start, " seconds")

search_start = time()
graph, full_path, total_cost, solution, cost_history = iter_search(search_factory, demands, ramping_data, 5)
search_stop = time()
println()

#largest = find_largest_time_period(12, demands)
#largest_model = build_and_optimize_largest_period(search_factory, demands[largest], ramping_data)
#loads = generate_random_loads(largest_model)
#scenarios = test_scenarios(search_factory, demands[largest], ramping_data, loads)
# Create a new model with fixed generator values from your graph solution
#=
verification_model = create_search_model(search_factory, 12, ramping_data, demands)
for t in keys(solution)
    for (gen, val) in solution[t]["generator_values"]
        fix(verification_model.model[:pg][t, gen], val, force=true)
    end
end
optimize!(verification_model.model)
status = termination_status(verification_model.model)
println("Graph solution feasibility: $status")
if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    println("Objective value: $(objective_value(verification_model.model))")
end
=#

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

savefig("cost_history.png")