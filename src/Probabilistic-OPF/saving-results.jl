using JuMP, Ipopt, Gurobi, Random
using PowerModels
using Statistics
using Distributions
import MathOptInterface as MOI
using CSV, DataFrames

include("probabilistic-OPF.jl")

function extract_case_name(file_path::String)
    # Extract the filename from the path
    filename = basename(file_path)
    # Remove the file extension
    case_name = replace(filename, r"\.m$" => "")
    return case_name
end

function run_parameter_sweep(
    file_path::String, 
    param_name::String,
    param_values::Vector,
    fixed_params::Dict,
    base_output_dir::String="./probabilistic-results"
)
    # Extract case name from file path
    case_name = extract_case_name(file_path)
    
    # Create structured output directory
    output_dir = joinpath(base_output_dir, case_name, param_name)
    mkpath(output_dir)
    
    # Create CSV file for results
    csv_path = joinpath(output_dir, "results.csv")
    
    # Get generator bus numbers from the data
    data = PowerModels.parse_file(file_path)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    
    # Get the bus numbers for each generator
    gen_buses = [gen["gen_bus"] for (i, gen) in gen_data]
    num_gens = length(gen_data)
    
    # Create a mapping from index to bus number
    gen_idx_to_bus = Dict()
    for (idx, (i, gen)) in enumerate(gen_data)
        gen_idx_to_bus[idx] = gen["gen_bus"]
    end
    
    println("Generator map for $case_name: $gen_idx_to_bus")
    
    # Prepare results dataframe
    results = DataFrame()
    results[!, param_name] = param_values
    results[!, :status] = fill("", length(param_values))
    results[!, :objective] = fill(NaN, length(param_values))
    
    # Add columns for each generator with the bus number
    for (idx, bus) in gen_idx_to_bus
        results[!, Symbol("pg_bus_$(bus)")] = fill(NaN, length(param_values))
    end
    
    # Run the parameter sweep
    for (i, param_value) in enumerate(param_values)
        println("Testing $case_name: $param_name = $param_value")
        
        # Set up parameters
        epsilon = fixed_params[:epsilon]
        confidence_level = fixed_params[:confidence_level]
        variation_type = fixed_params[:variation_type]
        variation_value = fixed_params[:variation_value]
        
        # Override the parameter being swept
        if param_name == "epsilon"
            epsilon = param_value
        elseif param_name == "confidence_level"
            confidence_level = param_value
        elseif param_name == "variation_value"
            variation_value = param_value
        end
        
        # Setup distributions
        distributions = setup_demand_distributions(file_path, variation_type, variation_value)
        
        # Create model with factory
        dc_factory = DCMPOPFModelFactory(file_path, Gurobi.Optimizer)
        
        try
            # Create and optimize model
            probabilistic_model = create_probabilistic_model(
                dc_factory, distributions, confidence_level, epsilon
            )
            
            optimize!(probabilistic_model.model)
            
            # Get status and results
            status = termination_status(probabilistic_model.model)
            
            if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
                results[i, :status] = "feasible"
                results[i, :objective] = objective_value(probabilistic_model.model)
                
                # Get pg values
                pg_vals = JuMP.value.(probabilistic_model.model[:pg])
                if ndims(pg_vals) == 1
                    # Single time period case
                    for (idx, bus) in gen_idx_to_bus
                        results[i, Symbol("pg_bus_$(bus)")] = pg_vals[idx]
                    end
                else
                    # Multiple time periods (use first period)
                    for (idx, bus) in gen_idx_to_bus
                        results[i, Symbol("pg_bus_$(bus)")] = pg_vals[1, idx]
                    end
                end
            else
                results[i, :status] = string(status)
            end
        catch e
            println("Error with $case_name: $param_name = $param_value: $e")
            results[i, :status] = "error"
        end
    end
    
    # Save results to CSV
    CSV.write(csv_path, results)
    println("$case_name $param_name tests complete. Results saved to $csv_path")
    
    return results
end

function run_all_parameter_sweeps(case_files::Vector{String}, output_dir::String="./probabilistic-results")
    # Define base parameters
    base_params = Dict(
        :epsilon => 6.0,
        :confidence_level => 0.90,
        :variation_type => :relative,
        :variation_value => 0.15
    )
    
    for file_path in case_files
        case_name = extract_case_name(file_path)
        println("Running parameter sweeps for $case_name...")
        
        # Run epsilon sweep
        # epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0]
        epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 
        1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 
        3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        run_parameter_sweep(file_path, "epsilon", epsilon_values, base_params, output_dir)
        
        # Run confidence level sweep
        # confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
        confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
        run_parameter_sweep(file_path, "confidence_level", confidence_levels, base_params, output_dir)
        
        # Run variation value sweep
        # variation_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        variation_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        run_parameter_sweep(file_path, "variation_value", variation_values, base_params, output_dir)
        
        println("All parameter sweeps complete for $case_name.")
    end
    
    println("All cases and parameter sweeps complete. Results saved to $output_dir")
end

# Example to run a single parameter sweep for multiple cases
function run_epsilon_sweep(case_files::Vector{String}, output_dir::String="./probabilistic-results")
    base_params = Dict(
        :epsilon => 1.0,  # Will be overridden
        :confidence_level => 0.95,
        :variation_type => :relative,
        :variation_value => 0.15
    )
    
    for file_path in case_files
        epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 
                         1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 
                         3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        run_parameter_sweep(file_path, "epsilon", epsilon_values, base_params, output_dir)
    end
end

function run_confidence_sweep(case_files::Vector{String}, output_dir::String="./probabilistic-results")
    base_params = Dict(
        :epsilon => 1.0,
        :confidence_level => 0.95,  # Will be overridden
        :variation_type => :relative,
        :variation_value => 0.15
    )
    
    for file_path in case_files
        confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
        run_parameter_sweep(file_path, "confidence_level", confidence_levels, base_params, output_dir)
    end
end

function run_variation_sweep(case_files::Vector{String}, output_dir::String="./probabilistic-results")
    base_params = Dict(
        :epsilon => 1.0,
        :confidence_level => 0.95,
        :variation_type => :relative,
        :variation_value => 0.15  # Will be overridden
    )
    
    for file_path in case_files
        variation_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        run_parameter_sweep(file_path, "variation_value", variation_values, base_params, output_dir)
    end
end

# Example usage with multiple case files
case_files = [
    "././Cases/case14.m",
    "././Cases/case300.m",
    "././Cases/case3.m",
    "././Cases/case5.m",
    "././Cases/case9.m",
    "././Cases/case118.m",
    "././Cases/case30.m",
    "././Cases/case18.m",
]

# Run all parameter sweeps for all cases
run_all_parameter_sweeps(case_files)

# Or run individual sweeps for all cases
# run_epsilon_sweep(case_files)
# run_confidence_sweep(case_files)
# run_variation_sweep(case_files)