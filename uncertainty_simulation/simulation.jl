using JuMP, Ipopt, Gurobi, Serialization, Random
using PowerModels, MPOPF, Statistics, LinearAlgebra
using DataFrames, CSV  # For data organization

# Test parameters
cases = ["case14"]
# mu_plus_values = [10.0, 50.0, 100.0, 500.0, 1000.0]

# Creating a loop that creates a list of values in an array 100 apart
mu_plus_values = [1000.0:1000.0:10000.0;]
mu_minus_values = [1000.0:1000.0:10000.0;]

# mu_minus_values = [10.0, 50.0, 100.0, 500.0, 1000.0]
variation_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
num_scenarios = 100

# Initialize results DataFrame
results = DataFrame(
    case = String[],
    sampling_method = String[],
    mu_plus = Float64[],
    mu_minus = Float64[],
    variation = Float64[],
    success_rate = Float64[]
)

# Run experiments
for case_name in cases
    file_path = "./Cases/$(case_name).m"
    
    for variation in variation_values
        # Statistical Sampling Method
        distributions = setup_demand_distributions(file_path, :relative, variation)
        
        for mu_plus in mu_plus_values
            for mu_minus in mu_minus_values
                mismatch_costs = (mu_plus, mu_minus)
                
                # Statistical sampling
                training_scenarios = sample_demand_scenarios(distributions, num_scenarios)
                dc_factory_Gurobi = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
                model_stat = create_model(dc_factory_Gurobi, training_scenarios, mismatch_costs)
                optimize_model(model_stat)
                PgValues_stat = JuMP.value.(model_stat.model[:pg])
                
                test_scenarios = sample_demand_scenarios(distributions, num_scenarios)
                success_rate_stat = test_concrete_solution(PgValues_stat, test_scenarios, dc_factory_Gurobi)
                
                push!(results, (case_name, "statistical", mu_plus, mu_minus, variation, success_rate_stat))
                
                # Covariance sampling
                training_scenarios_cov = generate_correlated_scenarios(file_path, num_scenarios, variation)
                model_cov = create_model(dc_factory_Gurobi, training_scenarios_cov, mismatch_costs)
                optimize_model(model_cov)
                PgValues_cov = JuMP.value.(model_cov.model[:pg])
                
                test_scenarios_cov = generate_correlated_scenarios(file_path, num_scenarios, variation)
                success_rate_cov = test_concrete_solution(PgValues_cov, test_scenarios_cov, dc_factory_Gurobi)
                
                push!(results, (case_name, "covariance", mu_plus, mu_minus, variation, success_rate_cov))
            end
        end
    end
end
results

# Save results
CSV.write("simulation_results.csv", results)
