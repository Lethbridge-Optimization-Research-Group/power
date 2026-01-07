#############################################################################
# Note: This file is curently being used for examples, nothing is permanent
#############################################################################

using JuMP, Ipopt, Gurobi, Serialization, Random
using PowerModels
using MPOPF
using Statistics
using CSV
using DataFrames




file_path = "./Cases/case300.m"
num_scenarios = 10
variation_value = 0.15
mismatch_costs::Tuple{Float64,Float64}=(100.0, 100.0)


# Get initial PG solution from first method
distributions = setup_demand_distributions(file_path, :relative, variation_value)
training_scenarios = sample_demand_scenarios(distributions, num_scenarios)

dc_factory_Gurobi = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
My_DC_model_Uncertainty = create_model(dc_factory_Gurobi, training_scenarios, mismatch_costs)
optimize_model(My_DC_model_Uncertainty)
# Output the final Pg Values
println("Final Pg values:")
println()
PgValues = JuMP.value.(My_DC_model_Uncertainty.model[:pg])
display(value.(My_DC_model_Uncertainty.model[:mu_plus]))
display(value.(My_DC_model_Uncertainty.model[:mu_minus]))


test_scenarios = sample_demand_scenarios(distributions, num_scenarios)
test_concrete_solution(PgValues, test_scenarios, dc_factory_Gurobi)

data = DataFrame(number = 1:10)
CSV.write("numbers.csv", data)




# Example of random scenarios
# --------------------------------------------------------------------------
file_path = "./Cases/case14.m"

base_loads = return_loads(file_path)
min = 0.25
max = 0.25

scenarios = get_random_scenarios(base_loads, min, max, 5)

display(scenarios)
# --------------------------------------------------------------------------






# Example for DC with new uncertainty functions for generating scenarios
# --------------------------------------------------------------------------
# file_path  = "./Cases/case14.m"
# distributions = setup_demand_distributions(file_path, :absolute, 0.15)
# scenarios = sample_demand_scenarios(distributions, 100, false)
# # Using DC Factory with Gurobi
# dc_factory_Gurobi = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
# My_DC_model_Uncertainty = create_model(dc_factory_Gurobi, scenarios)
# optimize_model(My_DC_model_Uncertainty)
# # Output the final Pg Values
# println("Final Pg values:")
# println()
# display(JuMP.value.(My_DC_model_Uncertainty.model[:pg]))
# display(JuMP.value.(My_DC_model_Uncertainty.model[:mu_plus]))
# display(JuMP.value.(My_DC_model_Uncertainty.model[:mu_minus]))
# --------------------------------------------------------------------------


#=
# --------------------------------------------------------------------------
# Example usage of local search w/ comparison against optimal model
# Initialize data files and variables
matpower_file_path = "./Cases/case300.m"
output_dir = "./Cases"
data = PowerModels.parse_file(matpower_file_path)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

results = []

# Generate the CSV used for auxilary ramping data
ramping_csv_file = generate_power_system_csv(data, output_dir)
# Parse the ramping data
ramping_data, demands = parse_power_system_csv(ramping_csv_file, matpower_file_path)
# Create factory for building model
search_factory = DCMPOPFSearchFactory(matpower_file_path, Ipopt.Optimizer)
# Initialize model with data and set num of time periods
search_model = create_search_model(search_factory, 12, ramping_data, demands)
# Optimize the model (this is the entire model, not decomposed. We can compare out search against this answer)
optimize!(search_model.model)

# Run a search on the same model and data as our optimal model
best_solution, best_cost, best_models, base_cost, current_demands, total_iterations = decomposed_mpopf_demand_search(search_factory, 12, ramping_data, demands)

push!(results, (objective_value(search_model.model), best_cost))

sum_of_opt = sum(x[1] for x in results)
sum_of_search = sum(x[2] for x in results)

println("Average difference: ")
println(sum_of_search / sum_of_opt)
# --------------------------------------------------------------------------
=#




# my_factory = DCMPOPFModelFactory("./Cases/case14.m", Gurobi.Optimizer)
# my_model = create_model(my_factory)
# optimize_model(my_model)


# factory = ACMPOPFModelFactory("./Cases/case14.m", Ipopt.Optimizer)

# data = PowerModels.parse_file(factory.file_path)
# PowerModels.standardize_cost_terms!(data, order=2)
# PowerModels.calc_thermal_limits!(data)

# ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
# bus = 1
# ref[:gen]
# ref[:bus]

# branch = ref[:branch][bus]
# va = value.(models["case14"]["AC"].model[:va])


# t = 1

# va_fr = va[t,branch["f_bus"]]
# va_to = va[t,branch["t_bus"]]
























# for case in keys(models)
# 	println("\n\n\nCase: ", case)
# 	for model in keys(models[case])
# 		println("Model: ", model)
# 		model = models[case][model].model

# 		# try
# 		# check feasibility of the constraints
# 		feasibility = find_infeasible_constraints(model)
# 		violations = find_bound_violations(model)

# 		# sum the violation differences
# 		total_violation = sum(getindex.(values(violations), 4))
# 		number_violations = length(values(violations))

# 		average_violation = total_violation / number_violations

# 		# look into the feasbility violations
# 		if length(feasibility) > 0
# 			for (con, val) in feasibility
# 				println("")
# 				println("Infeasible constraint: ", con)
# 				println("Current value: ", val)
# 				println("")
# 			end
# 		end

# 		# look into the bound violations below
# 		if length(violations) > 0
# 			println("Max violation: ", maximum(getindex.(values(violations), 4)))
# 			println("Average violation: ", average_violation)
# 		end
# 	end
# end

# Path to the case file
# file_path = "./Cases/case14.m"


# # Example for AC
# # --------------------------------------------------------------------------
# ac_factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)
# My_AC_model = create_model(ac_factory)
# optimize_model(My_AC_model)
# # --------------------------------------------------------------------------


# # Example for DC
# # --------------------------------------------------------------------------
# dc_factory = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
# My_DC_model = create_model(dc_factory)
# optimize_model(My_DC_model)
# # --------------------------------------------------------------------------


# # Multi Period Graphing Example for AC
# # --------------------------------------------------------------------------
# ac_factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)
# My_AC_model_Graphing_multi_period = create_model(ac_factory; time_periods=24, factors=[1.0, 1.05, 0.98, 1.03, 0.96, 0.97, 0.99, 1.0, 1.05, 1.03, 1.01, 0.95, 1.04, 1.02, 0.99, 0.99, 0.99, 0.95, 1.04, 1.02, 0.98, 1.0, 1.02, 0.97], ramping_cost=2000)
# optimize_model_with_plot(My_AC_model_Graphing_multi_period)
# # --------------------------------------------------------------------------


# # Example for DC with UncertaintyFactory
# # --------------------------------------------------------------------------
# file_path  = "./Cases/case14.m"
# load_scenarios_factors = generate_random_load_scenarios(5, 14)
# # Using DC Factory with Gurobi
# dc_factory_Gurobi = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
# My_DC_model_Uncertainty = create_model(dc_factory_Gurobi, load_scenarios_factors)
# optimize_model(My_DC_model_Uncertainty)
# # Output the final Pg Values
# println("Final Pg values:")
# println()
# display(JuMP.value.(My_DC_model_Uncertainty.model[:pg]))
# # --------------------------------------------------------------------------\
