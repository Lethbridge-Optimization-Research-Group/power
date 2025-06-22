"""
    generate_load_scenarios(num_scenarios::Int, num_buses::Int)

Generate load scenarios for a given number of scenarios and buses.
This method is outdated and depricated.

# Arguments
- `num_scenarios::Int`: The number of scenarios to generate.
- `num_buses::Int`: The number of buses in the case file.

# Returns
- `Dict{Int, Dict{Int, Float64}}`: A dictionary where each key 
is a scenario number and the value is another dictionary 
mapping bus numbers to load factors.
"""
function generate_random_load_scenarios(num_scenarios::Int, num_buses::Int)
    # Note: I made it so that the first scenario loads are defaults.
    load_scenarios_factors = Dict()
	bus_factors = Dict()
	for bus in 1:num_buses
		bus_factors[bus] = 1.0
	end
	load_scenarios_factors[1] = bus_factors
    for scenario in 2:num_scenarios
        scenario_factor = rand(0.95:0.01:1.05)
        bus_factors = Dict()
        for bus in 1:num_buses
            bus_factors[bus] = scenario_factor
        end
        load_scenarios_factors[scenario] = bus_factors
    end
    return load_scenarios_factors
end

"""
    setup_demand_distributions(file_path::String, variation_type::Symbol=:absolute, variation_value::Float64=0.15)

Setup demand distributions for a given case file.

# Arguments
- `file_path::String`: The path to the case file.
- `variation_type::Symbol`: The type of variation to apply (:absolute, :relative).
- `variation_value::Float64`: The value of the variation to apply (ex: 0.15).

# Returns
- `Dict()`: A dictionary where each key is a load number to a normal distribution.
"""
function setup_demand_distributions(file_path, variation_type::Symbol=:absolute, variation_value::Float64 = 0.15)
    data = PowerModels.parse_file(file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    distributions = Dict()
    
    for (i, load) in ref[:load]
        mu = load["pd"]
        if variation_type == :relative
            sigma = abs(mu) * variation_value
        elseif variation_type == :absolute
            sigma = variation_value
        else
            error("Invalid variation_type. Use :relative or :absolute")
        end
        distributions[i] = Normal(mu, sigma)
    end
    
    return distributions
end

function return_loads(file_path)
    data = PowerModels.parse_file(file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    loads = Dict()
    
    for (i, load) in ref[:load]
        loads[i] = load["pd"]
    end
    
    return loads
end

"""
    generate_correlated_scenarios(file_path::String, num_scenarios::Int64, variation_value::Float64=0.15)

Generate correlated load scenarios using Cholesky decomposition.

# Arguments
- `file_path::String`: Path to the case file
- `num_scenarios::Int64`: Number of scenarios to generate
- `variation_value::Float64`: Variation value for the standard deviation

# Returns
- Dictionary of scenarios with correlated load values
"""
function generate_correlated_scenarios(file_path::String, num_scenarios::Int64, variation_value::Float64=0.15)
    # Get base loads
    base_loads = return_loads(file_path)
    num_loads = length(base_loads)
    
    # Create base correlation matrix (example correlation structure)
    correlation_matrix = zeros(num_loads, num_loads)
    for i in 1:num_loads
        for j in 1:num_loads
            if i == j
                correlation_matrix[i,j] = 1.0
            else
                # Example: Correlation decreases with distance between load indices
                correlation_matrix[i,j] = 0.5^(abs(i-j))
            end
        end
    end
    
    # Convert correlation to covariance matrix
    stds = variation_value .* collect(values(base_loads))
    covariance_matrix = correlation_matrix .* (stds * stds')

    # Clamp negative values to 0 and ensure diagonal dominance
    covariance_matrix = max.(covariance_matrix, 0)

    # # Add small positive values to diagonal for stability
    for i in 1:num_loads
        covariance_matrix[i,i] += 1e-10
    end
    
    # Compute Cholesky decomposition
    c = cholesky(Symmetric(covariance_matrix)).L
    
    # Generate independent normal samples
    x = randn(num_loads, num_scenarios)
    
    # Convert to correlated samples
    y = c * x
    
    # Create scenarios dictionary
    scenarios = Dict()
    load_ids = collect(keys(base_loads))
    
    for s in 1:num_scenarios
        scenarios[s] = Dict()
        for (idx, load_id) in enumerate(load_ids)
            # Add mean (base load) to the correlated variation
            scenarios[s][load_id] = base_loads[load_id] + y[idx,s]
        end
    end
    
    return scenarios
end

"""
    verify_scenario_correlation(scenarios::Dict{Any,Dict{Any,Float64}}, expected_correlation::Matrix{Float64})

Verify that generated scenarios have the expected correlation structure.

# Arguments
- `scenarios`: Dictionary of generated scenarios
- `expected_correlation`: Expected correlation matrix

# Returns
- Nothing, prints correlation analysis
"""
function verify_scenario_correlation(scenarios::Dict{Any, Any}, expected_correlation::Matrix{Float64})
    # Calculate actual correlation from scenarios
    actual_correlation = calculate_load_correlations(scenarios)
    
    println("Correlation Analysis:")
    println("Maximum absolute difference between expected and actual correlation: ",
            maximum(abs.(expected_correlation - actual_correlation)))
    
    println("\nExpected Correlation Matrix:")
    display(expected_correlation)
    
    println("\nActual Correlation Matrix:")
    display(actual_correlation)
end

"""
    analyze_scenario_statistics(scenarios::Dict{Any,Dict{Any,Float64}}, base_loads::Dict{Any,Float64})

Analyze the statistical properties of generated scenarios.

# Arguments
- `scenarios`: Dictionary of generated scenarios
- `base_loads`: Dictionary of base load values

# Returns
- Nothing, prints statistical analysis
"""
function analyze_scenario_statistics(scenarios::Dict{Any, Any}, base_loads::Dict{Any, Any})
    num_scenarios = length(scenarios)
    load_ids = collect(keys(base_loads))
    
    println("\nScenario Statistics:")
    for load_id in load_ids
        values = [scenarios[s][load_id] for s in 1:num_scenarios]
        
        println("\nLoad $load_id:")
        println("  Base Value: $(base_loads[load_id])")
        println("  Mean: $(mean(values))")
        println("  Std Dev: $(std(values))")
        println("  Min: $(minimum(values))")
        println("  Max: $(maximum(values))")
    end
end

"""
    calculate_load_correlations(samples::Dict{Any,Dict{Any,Float64}})::Matrix{Float64}

Calculate correlation matrix between loads using provided samples.

# Arguments
- `samples`: Dictionary of sampled scenarios

# Returns
- Correlation matrix between loads
"""
function calculate_load_correlations(samples::Dict{Any, Any})::Matrix{Float64}
    # Get dimensions
    num_samples = length(samples)
    load_ids = collect(keys(samples[1]))
    num_loads = length(load_ids)
    
    # Convert samples to matrix form
    sample_matrix = zeros(num_samples, num_loads)
    for (s, scenario) in samples
        for (j, load_id) in enumerate(load_ids)
            sample_matrix[s,j] = scenario[load_id]
        end
    end
    
    return cor(sample_matrix)
end

"""
Tests given PG values against multiple scenarios, counting how many scenarios work.

# Arguments
- `pg_values`: DenseAxisArray of PG values
- `scenarios`: Dictionary of scenario dictionaries
- `file_path`: Path to the case file

# Returns
- Number of scenarios that successfully optimized with the given PG values
"""
function test_concrete_solution(pg_values::JuMP.Containers.DenseAxisArray, scenarios::Dict, dc_factory::AbstractMPOPFModelFactory)
    successful_scenarios = 0
    
    # Loop through each scenario
    for (s, scenario) in scenarios
        # Create new model with single scenario
        single_scenario = Dict(1 => scenario)
        model = create_model(dc_factory, single_scenario)
        
        # Fix the pg values - need to match the indices
        for g in axes(pg_values, 2)
            fix(model.model[:pg][1,g], pg_values[1,g]; force=true)
        end
        
        # Try to optimize
        try
            optimize!(model.model)
            if termination_status(model.model) == MOI.OPTIMAL
                # Check mu_plus values
                mu_plus_vals = JuMP.value.(model.model[:mu_plus])
                has_failures = false
                
                # Check if any mu_plus value is significantly greater than zero
                # Using tolerance of 1e-10 to avoid numerical issues
                for idx in eachindex(mu_plus_vals)
                    if mu_plus_vals[idx] > 1e-10
                        has_failures = true
                        break
                    end
                end
                
                if !has_failures
                    successful_scenarios += 1
                end
            end
        catch
            continue  # If optimization fails, move to next scenario
        end
    end
    
    return successful_scenarios
end

"""
    sample_demand_scenarios(distributions::Dict{Any, Any}, num_scenarios::Int64=1, debug::Bool=false)

Sample demand scenarios for a given set of distributions.

# Arguments
- `distributions::Dict{Any, Any}`: A dictionary where each key is a load number to a normal distribution.
- `num_scenarios::Int64`: The number of scenarios to sample.
- `debug::Bool`: A flag to print debug information.

# Returns
- `Dict{Int, Dict{Int, Float64}}`: A dictionary where each key is a scenario number and the value is another dictionary
mapping load numbers to sampled demand values.
"""
function sample_demand_scenarios(distributions::Dict{Any, Any}, num_scenarios::Int64=1, debug::Bool=false)
    scenarios = Dict()
    
    if debug
        println("Sampling diagnostics:")
        for (i, dist) in distributions
            println("Load $i: μ = $(mean(dist)), σ = $(std(dist))")
        end
        println()
    end
    
    for s in 1:num_scenarios
        scenario = Dict()
        for (i, dist) in distributions
            sampled_value = rand(dist)

            # Ensure non-negative demand
            # However it will in the long run shift the mean of the distribution
            # upwards. This is because the distribution is truncated at 0.
            # scenario[i] = max(0, sampled_value)
            # Here we will allow negative demand values.
            scenario[i] = sampled_value


            
            if debug
                println("Scenario $s, Load $i:")
                println("  Original Value: $(mean(dist))")
                println("  Original sample: $sampled_value")
                println("  After max(0, x): $(scenario[i])")
                println("  Z-score: $((sampled_value - mean(dist)) / std(dist))")
                println()
            end
        end
        scenarios[s] = scenario
    end
    
    if debug
        for (i, dist) in distributions
            samples = [scenarios[s][i] for s in 1:num_scenarios]
            println("Load $i statistics across all scenarios:")
            println("  Original Value: $(mean(dist))")
            println("  Mean: $(mean(samples))")
            println("  Std Dev: $(std(samples))")
            println("  Min: $(minimum(samples))")
            println("  Max: $(maximum(samples))")
            println()
        end
    end
    
    return scenarios
end

function set_model_uncertainty_variables!(power_flow_model::MPOPFModelUncertainty)
    model = power_flow_model.model
    T = power_flow_model.time_periods
    ref = PowerModels.build_ref(power_flow_model.data)[:it][:pm][:nw][0]
    scenarios = power_flow_model.scenarios

    @variable(model, mu_plus[t in 1:T, g in keys(ref[:gen]), s in 1:length(scenarios)] >= 0)
    @variable(model, mu_minus[t in 1:T, l in keys(ref[:bus]), s in 1:length(scenarios)] >= 0)
end

function set_model_uncertainty_objective_function!(power_flow_model::MPOPFModelUncertainty, factory::ACMPOPFModelFactory, mu_plus_cost::Float64, mu_minus_cost::Float64)
    model = power_flow_model.model
    data = power_flow_model.data
    T = power_flow_model.time_periods
    ramping_cost = power_flow_model.ramping_cost
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    pg = model[:pg]
    ramp_up = model[:ramp_up]
    ramp_down = model[:ramp_down]
    mu_plus = model[:mu_plus]
    mu_minus = model[:mu_minus]
    scenarios = power_flow_model.scenarios
    
    @objective(model, Min,
        sum(sum(gen_data[g]["cost"][1]*pg[t,g]^2 + gen_data[g]["cost"][2]*pg[t,g] + gen_data[g]["cost"][3] for g in keys(gen_data)) for t in 1:T) +
        sum(ramping_cost * (ramp_up[t, g] + ramp_down[t, g]) for g in keys(gen_data) for t in 2:T)
        # Adding some cost for mu_plus and mu_minus.
        + sum(10000 * (mu_plus[t, g, s]^2 + mu_minus[t, l, s]) for g in keys(ref[:gen]) for l in keys(ref[:load]) for t in 1:T for s in 1:length(scenarios))
    )
end

function set_model_uncertainty_constraints!(power_flow_model::MPOPFModelUncertainty, factory::ACMPOPFModelFactory)
    model = power_flow_model.model
    data = power_flow_model.data
    T = power_flow_model.time_periods
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    va = model[:va]
    p = model[:p]
    q = model[:q]
    pg = model[:pg]
    qg = model[:qg]
    vm = model[:vm]
    ramp_up = model[:ramp_up]
    ramp_down = model[:ramp_down]
    mu_plus = model[:mu_plus]
    mu_minus = model[:mu_minus]
    scenarios = power_flow_model.scenarios
    num_scenarios = length(scenarios)

    for t in 1:T
        for (i, bus) in ref[:ref_buses]
            @constraint(model, va[t,i] == 0)
        end

        for s in 1:num_scenarios
            scenario = scenarios[s]
            for b in keys(ref[:bus])
                
                # Active power balance at node i for scenario s
                bus_loads = [ref[:load][l] for l in ref[:bus_loads][b]]
                bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][b]]
    
                @constraint(model,
                    sum(p[t, a] for a in ref[:bus_arcs][b]) ==
                    sum(pg[t, g] + mu_plus[t, g, s] for g in ref[:bus_gens][b]) - 
                    sum(load["pd"] * scenario[b] + mu_minus[t, l, s] for load in bus_loads for l in ref[:bus_loads][b]) - 
                    sum(shunt["gs"] for shunt in bus_shunts)*vm[t,b]^2
                )
    
                @constraint(model,
                    sum(q[t, a] for a in ref[:bus_arcs][b]) ==
                    sum(qg[t, g] for g in ref[:bus_gens][b]) - 
                    sum(load["qd"] * scenario[b] + mu_minus[t, l, s] for load in bus_loads for l in ref[:bus_loads][b]) + 
                    sum(shunt["bs"] for shunt in bus_shunts)*vm[t,b]^2 
                )
            end
        end

        for (i, branch) in ref[:branch]
            f_idx = (i, branch["f_bus"], branch["t_bus"])
            t_idx = (i, branch["t_bus"], branch["f_bus"])

            p_to = p[t,t_idx]
            q_to = q[t,t_idx]
            p_fr = p[t,f_idx]
            q_fr = q[t,f_idx]

            va_fr = va[t,branch["f_bus"]]
            va_to = va[t,branch["t_bus"]]

            vm_fr = vm[t,branch["f_bus"]]
            vm_to = vm[t,branch["t_bus"]]

            g, b = PowerModels.calc_branch_y(branch)
            tr, ti = PowerModels.calc_branch_t(branch)
            ttm = tr^2 + ti^2

            g_fr = branch["g_fr"]
            b_fr = branch["b_fr"]
            g_to = branch["g_to"]
            b_to = branch["b_to"]

            @constraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
            @constraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )

            @constraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
            @constraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )

            @constraint(model, va_fr - va_to <= branch["angmax"])
            @constraint(model, va_fr - va_to >= branch["angmin"])

            @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
            @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
        end
    end

    for g in keys(gen_data)
        for t in 2:T
            @constraint(model, ramp_up[t, g] >= pg[t, g] - pg[t-1, g])
            @constraint(model, ramp_down[t, g] >= pg[t-1, g] - pg[t, g])
        end
    end
end

function set_model_uncertainty_objective_function!(power_flow_model::MPOPFModelUncertainty, factory::DCMPOPFModelFactory, mu_plus_cost::Float64, mu_minus_cost::Float64)
    model = power_flow_model.model
    data = power_flow_model.data
    T = power_flow_model.time_periods
    ramping_cost = power_flow_model.ramping_cost
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    pg = model[:pg]
    ramp_up = model[:ramp_up]
    ramp_down = model[:ramp_down]
    mu_plus = model[:mu_plus]
    mu_minus = model[:mu_minus]
    scenarios = power_flow_model.scenarios
    
    @objective(model, Min,
        sum(sum(gen_data[g]["cost"][1]*pg[t,g]^2 + gen_data[g]["cost"][2]*pg[t,g] + gen_data[g]["cost"][3] for g in keys(gen_data)) for t in 1:T) +
        sum(ramping_cost * (ramp_up[t, g] + ramp_down[t, g]) for g in keys(gen_data) for t in 2:T)
        # Adding some cost for mu_plus and mu_minus.
        # + sum(10000 * (mu_plus[t, g, s] + mu_minus[t, b, s]) for g in keys(ref[:gen]) for b in keys(ref[:bus]) for t in 1:T for s in 1:length(scenarios))
        + sum(mu_plus_cost * mu_plus[t, g, s] for t in 1:T for s in 1:length(scenarios) for g in keys(ref[:gen]))
        + sum(mu_minus_cost * mu_minus[t, b, s] for t in 1:T for s in 1:length(scenarios) for b in keys(ref[:bus]))
    )
end

function set_model_uncertainty_constraints!(power_flow_model::MPOPFModelUncertainty, factory::DCMPOPFModelFactory)
    model = power_flow_model.model
    data = power_flow_model.data
    T = power_flow_model.time_periods
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    va = model[:va]
    p = model[:p]
    pg = model[:pg]
    ramp_up = model[:ramp_up]
    ramp_down = model[:ramp_down]
    mu_plus = model[:mu_plus]
    mu_minus = model[:mu_minus]
    scenarios = power_flow_model.scenarios
    num_scenarios = length(scenarios)

    p_expr = Dict()
    for t in 1:T
        p_expr[t] = Dict()
    end
    # Iterate over each time period
    for t in 1:T
        p_expr[t] = Dict([((l, i, j), 1.0 * p[t, (l, i, j)]) for (l, i, j) in ref[:arcs_from]])
        p_expr[t] = merge(p_expr[t], Dict([((l, j, i), -1.0 * p[t, (l, i, j)]) for (l, i, j) in ref[:arcs_from]]))
    end

    for t in 1:T
        for (i, bus) in ref[:ref_buses]
            @constraint(model, va[t,i] == 0)
        end

        for s in 1:num_scenarios
            scenario = scenarios[s]
            for (b, bus) in ref[:bus]
                
                # Active power balance at node i for scenario s
                # This is the original bus_loads calculation
                # bus_loads = [ref[:load][l] for l in ref[:bus_loads][b]]
                bus_loads = ref[:bus_loads][b]
                bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][b]]

                # This is the original constraint
                # @constraint(model,
                #     sum(p_expr[t][a] for a in ref[:bus_arcs][b]) ==
                #     sum(pg[t, g] + mu_plus[t, g, s] for g in ref[:bus_gens][b]) - 
                #     sum(load["pd"] for load in bus_loads) - 
                #     sum(shunt["gs"] for shunt in bus_shunts)*1.0^2 - mu_minus[t, b, s]
                # )

                @constraint(model,
                    sum(p_expr[t][a] for a in ref[:bus_arcs][b]) ==
                    sum(pg[t, g] + mu_plus[t, g, s] for g in ref[:bus_gens][b]) - 
                    sum(scenario[l] for l in bus_loads) - 
                    sum(shunt["gs"] for shunt in bus_shunts)*1.0^2 - mu_minus[t, b, s]
                )

            end
        end

        for (i,branch) in ref[:branch]
            f_idx = (i, branch["f_bus"], branch["t_bus"])
    
            p_fr = p[t,f_idx]
    
            va_fr = va[t,branch["f_bus"]]
            va_to = va[t,branch["t_bus"]]
    
            g, b = PowerModels.calc_branch_y(branch)
    
            @constraint(model, p_fr == -b*(va_fr - va_to))
        
            @constraint(model, va_fr - va_to <= branch["angmax"])
            @constraint(model, va_fr - va_to >= branch["angmin"])
        end
    end

    for g in keys(gen_data)
        for t in 2:T
            @constraint(model, ramp_up[t, g] >= pg[t, g] - pg[t-1, g])
            @constraint(model, ramp_down[t, g] >= pg[t-1, g] - pg[t, g])
        end
    end
end
