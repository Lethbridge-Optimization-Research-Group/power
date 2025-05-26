using JuMP, Ipopt, Gurobi, Serialization, Random, Graphs, MetaGraphs, MathOptInterface
using PowerModels
using MPOPF
using Statistics, Plots, GraphRecipes

include("graph_search.jl")
include("search_functions.jl")

matpower_file_path = "./Cases/case14.m"
#matpower_file_path = "./Cases/case300.m"
#matpower_file_path = "./Cases/case1354pegase.m"

output_dir = "./Cases"
data = PowerModels.parse_file(matpower_file_path)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

#TODO: Try and reuse models 
#TODO: Tweak search parameters, adjust variation
#TODO: See if we can use powermodels for PF 
#TODO: Calculate cost manually, and only test scenarios w lower cost
#TODO: Look into running in parallel
#TODO: see if we can silence PowerModels output as well
#TODO: generate Pg values in a smart way, avoid random
    #TODO: Compare ramping and Pg of search model to the optimal model
    #TODO: Check pmax and pmin before testing feasibility 
        # virtually 0% of failed cases are due to pmin / pmax violation
    #TODO: Count how many of the infeasible scenarios are due to pmin/pmax violations
        # Done
#TODO: OR generate values inside of pmax/pmin and compare graphs
#TODO: Increase ramping cost and compare to optimal
    #TODO: Parellelization (generate shortest path, test each node in parallel)
#TODO: Hot start feasibility testing with current solution values 
#TODO: Modify for AC and see how much time / solution quality increases

# Save input data (csv) for a baseline for each case

# FUTURE
#TODO: Develop Newton method for solving PF and compare time with
# a warm start to JuMP model


differences = Float64[]
#for i in 1:1 =#
    ramping_csv_file = generate_power_system_csv(data, output_dir)
    ramping_data, demands = parse_power_system_csv("./Cases/MPOPF_Cases/111/case14_111.csv", matpower_file_path)

    search_factory = DCMPOPFSearchFactory(matpower_file_path, Gurobi.Optimizer)
    search_model = create_search_model(search_factory, 10, ramping_data, demands)
    opt_start = time()
    optimize!(search_model.model)
    opt_stop = time()
    println(opt_stop - opt_start, " seconds")

    search_start = time()
    info = iter_search(data, search_factory, demands, ramping_data, 10)
    search_stop = time()
    println()

    opt = objective_value(search_model.model)
    diff = info[:cost] / opt
    push!(differences, diff)
#end 
println("Seconds: ", search_stop - search_start)
println("Difference: ", info[:cost] / opt) 
#println(sum(differences) / 10)
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
#=
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

#=
case14 - 6.36%
case18 - 2.91%
case30Q - 3.14%
case39 - 5.53%
case118 - 1.13%
case300 - ~ 1%
case1354 - 20 iter - 2.34%
=#
#=
# look into creating a custom model, and then calling
# PowerModels.solve_model() on it after variables are fixed
data["gen"]["1"]["pg"] = 2.15606
data["gen"]["2"]["pg"] = 0.373197
data["gen"]["3"]["pg"] = 0.0206484
data["gen"]["4"]["pg"] = 0.0203327
data["gen"]["5"]["pg"] = 0.0198465

#test_model = PowerModels.solve_pf(data, DCPPowerModel, Gurobi.Optimizer)
=#

#test = get_generation_and_ramping_costs(info, search_model)