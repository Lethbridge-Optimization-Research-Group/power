#############################################################################
# Note: This file is currently being used for examples, nothing is permanent
#############################################################################

# using JuMP, Ipopt, Gurobi, Serialization, Random
# using PowerModels
# using MPOPF
# using Gurobi
# using JuMP
# using Statistics
# using CSV
# using DataFrames



################################################################################
# Uncertainty Examples
################################################################################



using MPOPF
using Gurobi
using JuMP

file_path = "./Cases/case14.m"


# Example 1: Using the existing scenario-based approach
num_scenarios = 10
variation_value = 0.15
mismatch_costs::Tuple{Float64,Float64}=(10000000.0, 10000000.0)

distributions = setup_demand_distributions(file_path, :relative, variation_value)
scenarios = sample_demand_scenarios(distributions, num_scenarios)

dc_factory_Gurobi = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)

DC_model_scenario = create_model(dc_factory_Gurobi, scenarios, mismatch_costs)
optimize!(DC_model_scenario.model)

optimal_cost = objective_value(DC_model_scenario.model)
println("Optimal Cost: ", optimal_cost)
println()
# Output the final Pg Values
println("Final Pg values:")
println()
PgValues = JuMP.value.(DC_model_scenario.model[:pg])
display(PgValues)
println()
display(value.(DC_model_scenario.model[:mu_plus]))
println()
display(value.(DC_model_scenario.model[:mu_minus]))
println()



# Example 2: Using the new chance-constrained approach
confidence_level=0.95
epsilon=1.0
variation_type=:relative
variation_value=0.15

distributions = setup_demand_distributions(file_path, variation_type, variation_value)

dc_factory = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)

model_cc = create_model(dc_factory, distributions, confidence_level, epsilon)
optimize!(model_cc.model)

optimal_cost = objective_value(model_cc.model)
println("Optimal Cost: ", optimal_cost)
println()
# Output the final Pg Values
pg_values = JuMP.value.(model_cc.model[:pg])
println("Optimal Generation Values:")
display(pg_values)
println()



# Example 3: Using the unified interface

# Scenario-based
num_scenarios = 10
variation_value = 0.15
mismatch_costs::Tuple{Float64,Float64}=(10000000.0, 10000000.0)
dc_factory = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
distributions = setup_demand_distributions(file_path, :relative, variation_value)
scenarios = sample_demand_scenarios(distributions, num_scenarios)

model_s = create_model_with_uncertainty(dc_factory, ScenarioBased,
    scenarios = scenarios,
    mismatch_costs = mismatch_costs
)
optimize!(model_s.model)
optimal_cost_s = objective_value(model_s.model)
println("Optimal Cost (Scenario-based): ", optimal_cost_s)
println()
# Output the final Pg Values
println("Final Pg values:")
println()
PgValues = JuMP.value.(model_s.model[:pg])
display(PgValues)
println()
display(value.(model_s.model[:mu_plus]))
println()
display(value.(model_s.model[:mu_minus]))
println()

# Chance-constrained
confidence_level=0.95
epsilon=1.0
variation_type=:relative
variation_value=0.15

distributions = setup_demand_distributions(file_path, variation_type, variation_value)

model_c = create_model_with_uncertainty(dc_factory, ChanceConstrained,
    distributions = distributions,
    confidence_level = confidence_level,
    epsilon = epsilon
)
optimize!(model_c.model)

optimal_cost_c = objective_value(model_c.model)
println("Optimal Cost (Chance-constrained): ", optimal_cost_c)
println()
# Output the final Pg Values
pg_values = JuMP.value.(model_c.model[:pg])
println("Optimal Generation Values (Chance-constrained):")
display(pg_values)
println()



################################################################################
# End of Uncertainty Examples
################################################################################











# # Example for DC with uncertainty functions for generating scenarios
# file_path = "./Cases/case14.m"
# num_scenarios = 10
# variation_value = 0.15
# mismatch_costs::Tuple{Float64,Float64}=(100.0, 100.0)


# # Get initial PG solution from first method
# distributions = setup_demand_distributions(file_path, :relative, variation_value)
# training_scenarios = sample_demand_scenarios(distributions, num_scenarios)

# dc_factory_Gurobi = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
# My_DC_model_Uncertainty = create_model(dc_factory_Gurobi, training_scenarios, mismatch_costs)
# optimize_model(My_DC_model_Uncertainty)
# # Output the final Pg Values
# println("Final Pg values:")
# println()
# PgValues = JuMP.value.(My_DC_model_Uncertainty.model[:pg])
# display(value.(My_DC_model_Uncertainty.model[:mu_plus]))
# display(value.(My_DC_model_Uncertainty.model[:mu_minus]))


# test_scenarios = sample_demand_scenarios(distributions, num_scenarios)
# test_concrete_solution(PgValues, test_scenarios, dc_factory_Gurobi)


# Example for DC with uncertainty functions for generating scenarios
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


# # Example for DC with UncertaintyFactory (deprecated)
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
