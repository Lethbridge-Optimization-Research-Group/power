using JuMP, PowerModels, Ipopt, Random
using MPOPF

"""
    create_initial_feasible_solution(data, time_periods, demands, ramping_data)

Set all generator ouputs to half of total capacity and adjusts to meet demands
of the model

"""
function create_initial_feasible_solution(data, time_periods, demands, ramping_data)
    solution = []
    for t in 1:time_periods
        pg = Dict()
        # Initialize generators to their midpoints
        for (i, gen) in data["gen"]
            gen_id = parse(Int, i)
            pg[gen_id] = (gen["pmin"] + gen["pmax"]) / 2
        end
        # Adjust to meet demand
        pg = adjust_to_meet_demand(Dict(t => pg), data, Dict(t => demands[t]), t, ramping_data)[t]
        push!(solution, pg)
    end
    return solution
end


function check_solution(solution, data, time_periods, ramping_data, demands, factory)
    models = []
    time_periods = length(solution)
    for t in 1:time_periods
        model = create_search_model(factory, 1, ramping_data, [demands[t]])
        
        # Add ramping constraints to the model
        if t > 1
            for (i, _) in data["gen"]
                gen_id = parse(Int, i)
                prev_pg = value(models[t-1].model[:pg][1,gen_id])
                ramp_limit = ramping_data["ramp_limits"][gen_id]
                
                @constraint(model.model, model.model[:pg][1,gen_id] <= prev_pg + ramp_limit)
                @constraint(model.model, model.model[:pg][1,gen_id] >= prev_pg - ramp_limit)
            end
        end

        # Set initial values
        for (i, pg) in solution[t]
            set_start_value(model.model[:pg][1,i], pg)
        end

        optimize!(model.model) # optimize!(model.model)

    end

    if !models.isempty()
        if models.length() == time_periods
            return true
        end
    end
    return false
end





      #= 
        # Check if the optimization was solved
        if termination_status(model.model) != MOI.OPTIMAL && termination_status(model.model) != MOI.LOCALLY_SOLVED
            println("Warning: Optimization for time period $t failed with status $(termination_status(model.model))")
            return solution, models  # Return the original solution if optimization fails
        end
        
        push!(models, model)
    end
    
    new_solution = [Dict(i => value(model.model[:pg][1,i]) for i in keys(solution[t])) for (t, model) in enumerate(models)]
    
    return new_solution, models
end
=#
"""
    adjust_to_meet_demand(solution, data, demands, t, ramping_data)

Modify outputs to ensure demand for time periods is met

"""
function adjust_to_meet_demand(solution, data, demands, t, ramping_data)
    total_generation = sum(values(solution[t]))
    total_demand = sum(demands[t])
    diff = total_demand - total_generation

    if diff <= 1e-6  # If difference is negligible or negative, consider it met
        return solution
    end

    # Sort generators by their cost efficiency (considering both generation and ramping costs)
    sorted_gens = sort(collect(data["gen"]),
                       by=x->(x[2]["cost"][2] + ramping_data["costs"][parse(Int, x[1])]),
                       rev=true)

    for (i, gen) in sorted_gens
        gen_id = parse(Int, i)
        ramp_limit = ramping_data["ramp_limits"][gen_id]

        # Ramp up by the difference or ramping limit, whichever is smaller
        max_increase = min(gen["pmax"] - solution[t][gen_id], ramp_limit)
        increase = min(diff, max_increase)
        solution[t][gen_id] += increase
        diff -= increase

        if abs(diff) < 1e-6
            break
        end
    end

    return solution
end

"""
    apply_ramping_constraints(solution, ramping_data, t)

Modify generator outputs to ensure ramping constraints are followed

"""
function apply_ramping_constraints(solution, ramping_data, t)
    if t > 1
        for (gen_id, current_output) in solution[t]
            prev_output = solution[t-1][gen_id]
            ramp_limit = ramping_data["ramp_limits"][gen_id]
            
            if current_output > prev_output
                solution[t][gen_id] = min(current_output, prev_output + ramp_limit)
            else
                solution[t][gen_id] = max(current_output, prev_output - ramp_limit)
            end
        end
    end
    return solution
end

"""
    optimize_solution(solution, data, ramping_data, demands, factory)

Optimize individual time periods and check for feasibility/termination status

"""
function optimize_solution(solution, data, ramping_data, demands, factory)
    models = []
    time_periods = length(solution)
    for t in 1:time_periods
        model = create_search_model(factory, 1, ramping_data, [demands[t]])
        
        # Add ramping constraints to the model
        if t > 1
            for (i, _) in data["gen"]
                gen_id = parse(Int, i)
                prev_pg = value(models[t-1].model[:pg][1,gen_id])
                ramp_limit = ramping_data["ramp_limits"][gen_id]
                
                @constraint(model.model, model.model[:pg][1,gen_id] <= prev_pg + ramp_limit)
                @constraint(model.model, model.model[:pg][1,gen_id] >= prev_pg - ramp_limit)
            end
        end

        # Set initial values
        for (i, pg) in solution[t]
            set_start_value(model.model[:pg][1,i], pg)
        end

        optimize!(model.model) # optimize!(model.model)
        
        # Check if the optimization was solved
        if termination_status(model.model) != MOI.OPTIMAL && termination_status(model.model) != MOI.LOCALLY_SOLVED
            println("Warning: Optimization for time period $t failed with status $(termination_status(model.model))")
            return solution, models  # Return the original solution if optimization fails
        end
        
        push!(models, model)
    end
    
    new_solution = [Dict(i => value(model.model[:pg][1,i]) for i in keys(solution[t])) for (t, model) in enumerate(models)]
    
    return new_solution, models
end

"""
    calculate_total_cost(solution, models, ramping_data)

Sum costs of individual time periods, calculate ramping costs 
between time periods and return total

"""
function calculate_total_cost(solution, models, ramping_data)
    time_periods = length(solution)
    operation_cost = sum(objective_value(model.model) for model in models)
    ramping_cost = sum(
        ramping_data["costs"][gen_id] * abs(solution[t][gen_id] - solution[t-1][gen_id])
        for t in 2:time_periods
        for gen_id in keys(solution[t])
    )
    return operation_cost + ramping_cost
end

"""
    is_feasible_solution(models)

Checks models for feasibility

"""
function is_feasible_solution(models)
    return all(termination_status(model.model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] for model in models)
end

function decomposed_mpopf_local_search(factory, time_periods, ramping_data, demands; max_iterations=1000, max_time=10)
    data = PowerModels.parse_file(factory.file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    
    # Create base solution to then begin search from
    solution = create_initial_feasible_solution(data, time_periods, demands, ramping_data)
    best_solution, best_models = optimize_solution(solution, data, ramping_data, demands, factory)
    best_cost = calculate_total_cost(best_solution, best_models, ramping_data)
    base_cost = best_cost

    current_solution = best_solution
    current_cost = best_cost
    current_models = best_models

    start_time = time()
    no_improvement_count = 0
    total_iterations = 0

    while total_iterations < max_iterations
        # Sets a time limit for the search
        if time() - start_time > max_time
            println("Time limit reached. Stopping search.")
            return best_solution, best_cost, best_models, base_cost, total_iterations
        end
        total_iterations += 1
        # Choose a random time period to adjust
        t = rand(1:time_periods)

        # Choose a random generator
        gen_ids = keys(data["gen"])
        selected_gen_id = rand(gen_ids)
        
        # Make a random adjustment
        new_solution = deepcopy(current_solution)

        gen_id = parse(Int, selected_gen_id)
        ramp_limit = ramping_data["ramp_limits"][gen_id]

        if t > 1
            prev_output = new_solution[t-1][gen_id]
            max_down = max(data["gen"]["$gen_id"]["pmin"], prev_output - ramp_limit) - new_solution[t][gen_id]
            max_up = min(data["gen"]["$gen_id"]["pmax"], prev_output + ramp_limit) - new_solution[t][gen_id]
        else
            max_down = data["gen"]["$gen_id"]["pmin"] - new_solution[t][gen_id]
            max_up = data["gen"]["$gen_id"]["pmax"] - new_solution[t][gen_id]
        end
        
        adjustment = rand(max_down:0.01:max_up)
        new_solution[t][gen_id] += adjustment
        
        # Adjust demand for affected time periods
        for i in t:time_periods
            new_solution = adjust_to_meet_demand(new_solution, data, demands, i, ramping_data)
        end
        
        new_solution, new_models = optimize_solution(new_solution, data, ramping_data, demands, factory)
        new_cost = calculate_total_cost(new_solution, new_models, ramping_data)
        
        if new_cost < best_cost && is_feasible_solution(new_models)
            best_solution = new_solution
            best_models = new_models
            best_cost = new_cost
            current_solution = new_solution
            current_cost = new_cost
            current_models = new_models
            no_improvement_count = 0
        else 
            no_improvement_count += 1
        end
        
        if no_improvement_count >= 50
            println("No improvement after $no_improvement_count iterations. Stopping search.")
            return best_solution, best_cost, best_models, base_cost, total_iterations
        end

    end
    
    return best_solution, best_cost, best_models, base_cost, total_iterations
end


function decomposed_mpopf_demand_search(factory, time_periods, ramping_data, demands; max_iterations=1000, max_time=300, demand_step=0.01)
    data = PowerModels.parse_file(factory.file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    
    solution = create_initial_random_solution(data, time_periods, demands, ramping_data, factory)
    best_solution, best_models = optimize_solution(solution, data, ramping_data, demands, factory)
    best_cost = calculate_total_cost(best_solution, best_models, ramping_data)
    base_cost = best_cost

    current_solution = best_solution
    current_cost = best_cost
    current_models = best_models
    current_demands = deepcopy(demands)

    start_time = time()
    no_improvement_count = 0
    escape_attempts = 0

    # Find the time period with maximum total demand (demand does not need to be increased)
    max_demand_period = argmax([sum(d) for d in demands])
    max_total_demand = sum(demands[max_demand_period])

    total_iterations = 0

    for iteration in 1:max_iterations
        if time() - start_time > max_time
            println("Time limit reached. Stopping search.")
            break
        end
        total_iterations += 1
        # Adjsut random time period excluding max demand period
        available_periods = setdiff(1:time_periods, max_demand_period)
        t = rand(available_periods)
        
        # Slightly increase the demand for time period
        new_demands = deepcopy(current_demands)
        for (i, load) in enumerate(new_demands[t])
            new_load = min(load + demand_step, demands[max_demand_period][i])
            new_demands[t][i] = new_load
        end
        
        # Check we don't exceed the maximum total demand
        if sum(new_demands[t]) > max_total_demand
            scale_factor = max_total_demand / sum(new_demands[t])
            new_demands[t] = new_demands[t] .* scale_factor
        end
        
        # Create a new solution based on the adjusted demand
        new_solution = adjust_to_meet_demand(current_solution, data, new_demands, t, ramping_data)
        new_solution, new_models = optimize_solution(new_solution, data, ramping_data, new_demands, factory)
        new_cost = calculate_total_cost(new_solution, new_models, ramping_data)
        
        if (new_cost < best_cost && is_feasible_solution(new_models)) && length(new_models) == time_periods && check_demands_met(new_solution, demands)
            best_solution = new_solution
            best_models = new_models
            best_cost = new_cost
            current_solution = new_solution
            current_cost = new_cost
            current_models = new_models
            current_demands = new_demands
            no_improvement_count = 0
            
        else
            no_improvement_count += 1
        end

        if escape_attempts < 5

        end
        
        if no_improvement_count >= 10
            break
        end
    end
    
    return best_solution, best_cost, best_models, base_cost, current_demands, total_iterations
end


"""
    check_ramping_limits(solution, ramping_data)

Checks a solved model for adherence to ramping constraints.
Identifies which time periods/gen violate constraints

"""
function check_ramping_limits(solution, ramping_data)
    time_periods = length(solution)
    violations = []

    for t in 2:time_periods
        for (gen_id, current_output) in solution[t]
            prev_output = solution[t-1][gen_id]
            ramp_limit = ramping_data["ramp_limits"][gen_id]
            
            ramp_up = current_output - prev_output
            ramp_down = prev_output - current_output
            
            if ramp_up > ramp_limit + 1e-6  # Adding small tolerance for floating-point errors
                push!(violations, (t, gen_id, "up", ramp_up, ramp_limit))
            elseif ramp_down > ramp_limit + 1e-6
                push!(violations, (t, gen_id, "down", ramp_down, ramp_limit))
            end
        end
    end

    if isempty(violations)
        println("All ramping limits are respected.")
        return true
    else
        println("Ramping limit violations found:")
        for (t, gen_id, direction, actual, limit) in violations
            println("Time $t, Generator $gen_id: $direction ramp of $actual exceeds limit $limit by ", actual - limit)
        end
        return false
    end
end


"""
    check_demands_met(solution, initial_demands, tolerance=1e-6)

Checks solved model to see if all minimum demands are met.
Identifies which time periods are not having demands met

"""
function check_demands_met(solution, initial_demands, tolerance=1e-6)
    time_periods = length(solution)
    violations = []

    for t in 1:time_periods
        total_generation = sum(values(solution[t]))
        total_demand = sum(initial_demands[t])
        
        if total_generation < total_demand - tolerance
            push!(violations, (t, total_generation, total_demand))
        end
    end

    if isempty(violations)
        return true
    end
    return false
end

function check_demands_met_output(solution, initial_demands, tolerance=1e-6)
    time_periods = length(solution)
    violations = []

    for t in 1:time_periods
        total_generation = sum(values(solution[t]))
        total_demand = sum(initial_demands[t])
        
        if total_generation < total_demand - tolerance
            push!(violations, (t, total_generation, total_demand))
        end
    end

    if isempty(violations)
        println("All demands are met within tolerance.")
        return true
    else
        println("Demand mismatches found:")
        for (t, gen, demand) in violations
            mismatch = gen - demand
            println("Time $t: Generation ($gen) - Demand ($demand) = $mismatch")
        end
        return false
    end
end


function sort_time_periods_by_demand(demands)
    return sortperm(sum.(demands), rev=true)
end

function set_initial_generator_output(model, t, demands, gen_data)
    total_demand = sum(demands[t])
    total_capacity = sum(gen["pmax"] for (_, gen) in gen_data)
    
    if total_demand > total_capacity
        error("Insufficient generation capacity for time period $t")
    end
    
    # Distribute demand proportionally among generators
    for (g, gen) in gen_data
        set_start_value(model[:pg][t, g], (gen["pmax"] / total_capacity) * total_demand)
    end
end

function adjust_adjacent_periods(model, t, adjacent_t, gen_data, ramp_data, direction)
    for (g, gen) in gen_data
        current_output = value(model[:pg][t, g])
        ramp_limit = ramp_data["ramp_limits"][parse(Int, g)]
        
        if direction == :before
            new_output = max(gen["pmin"], current_output - ramp_limit)
        else # :after
            new_output = max(gen["pmin"], current_output + ramp_limit)
        end
        
        set_start_value(model[:pg][adjacent_t, g], new_output)
    end
end

function create_decomposed_mpopf_model(factory, time_periods, ramping_data, demands)
    data = PowerModels.parse_file(factory.file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    
    model = JuMP.Model(factory.optimizer)
    power_flow_model = MPOPFSearchModel(model, data, time_periods, ramping_data, demands)
    
    set_model_variables!(power_flow_model, factory)
    set_model_objective_function!(power_flow_model, factory)
    set_model_constraints!(power_flow_model, factory)
    
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    
    sorted_periods = sort_time_periods_by_demand(demands)
    
    for t in sorted_periods
        set_initial_generator_output(model, t, demands, gen_data)
        
        if t > 1
            adjust_adjacent_periods(model, t, t-1, gen_data, ramping_data, :before)
        end
        if t < time_periods
            adjust_adjacent_periods(model, t, t+1, gen_data, ramping_data, :after)
        end
        
        # Optimize the current time period
        @objective(model, Min, sum(gen_data[g]["cost"][1]*model[:pg][t,g]^2 + 
                                   gen_data[g]["cost"][2]*model[:pg][t,g] + 
                                   gen_data[g]["cost"][3] for g in keys(gen_data)))
        optimize!(model)
        
        # Check if optimization reduced power generation
        for (g, gen) in gen_data
            if value(model[:pg][t, g]) < start_value(model[:pg][t, g])
                # If reduced, set it back to the initial value
                fix(model[:pg][t, g], start_value(model[:pg][t, g]))
            end
        end
    end
    
    # Set the full objective function for the final optimization
    set_model_objective_function!(power_flow_model, factory)
    optimize!(model)
    
    return power_flow_model
end


"""
    create_initial_random_solution(data, time_periods, demands, ramping_data, factory; 
                                 max_attempts=100, time_limit=60)

Generate a random feasible initial solution for the power system optimization problem.
Returns nil if no feasible solution is found within the specified attempts/time limit.
"""
function create_initial_random_solution(data, time_periods, demands, ramping_data, factory;
                                      max_attempts=100, time_limit=60)
    start_time = time()
    max_attempts = 10000000
    for attempt in 1:max_attempts
        if time() - start_time > time_limit
            println("Time limit reached after $attempt attempts")
            return nothing
        end
        
        # Generate random solution while respecting basic bounds
        solution = generate_bounded_random_solution(data, time_periods)
        
        # Adjust for demands and ramping constraints
        solution = adjust_solution_for_constraints(solution, data, demands, ramping_data, time_periods)
        
        # Check if solution is feasible
        if verify_solution_feasibility(solution, data, time_periods, ramping_data, demands, factory)
            println("Found feasible random solution after $attempt attempts")
            return solution
        end
    end
    
    error("Failed to find feasible solution after $max_attempts attempts")
    return nothing
end

"""
    generate_bounded_random_solution(data, time_periods)

Generate random solution respecting generator bounds and total demand requirements.
"""
function generate_bounded_random_solution(data, time_periods)
    solution = []
    
    for t in 1:time_periods
        pg = Dict()
        total_demand = 0
        
        # First pass: generate random values within bounds
        for (i, gen) in data["gen"]
            gen_id = parse(Int, i)
            pg[gen_id] = gen["pmin"] + rand() * (gen["pmax"] - gen["pmin"])
            total_demand += pg[gen_id]
        end
        
        push!(solution, pg)
    end
    
    return solution
end

"""
    adjust_solution_for_constraints(solution, data, demands, ramping_data, time_periods)

Adjust the random solution to better meet system constraints before verification.
"""
function adjust_solution_for_constraints(solution, data, demands, ramping_data, time_periods)
    adjusted_solution = deepcopy(solution)
    
    # First adjust for demand
    for t in 1:time_periods
        adjusted_solution = adjust_to_meet_demand(adjusted_solution, data, demands, t, ramping_data)
    end
    
    # Then adjust for ramping
    for t in 2:time_periods
        for (i, gen) in data["gen"]
            gen_id = parse(Int, i)
            prev_output = adjusted_solution[t-1][gen_id]
            current_output = adjusted_solution[t][gen_id]
            ramp_limit = ramping_data["ramp_limits"][gen_id]
            
            # Adjust if ramping constraint is violated
            if abs(current_output - prev_output) > ramp_limit
                if current_output > prev_output
                    adjusted_solution[t][gen_id] = prev_output + ramp_limit
                else
                    adjusted_solution[t][gen_id] = prev_output - ramp_limit
                end
            end
        end
        # Readjust for demand after ramping constraints
        adjusted_solution = adjust_to_meet_demand(adjusted_solution, data, demands, t, ramping_data)
    end
    
    return adjusted_solution
end

"""
    verify_solution_feasibility(solution, data, time_periods, ramping_data, demands, factory)

Verify if a solution is feasible by checking all constraints and running a test optimization.
"""
function verify_solution_feasibility(solution, data, time_periods, ramping_data, demands, factory)
    # Check basic bounds
    for t in 1:time_periods
        for (i, gen) in data["gen"]
            gen_id = parse(Int, i)
            if solution[t][gen_id] < gen["pmin"] || solution[t][gen_id] > gen["pmax"]
                return false
            end
        end
    end
    
    # Check ramping constraints
    for t in 2:time_periods
        for (i, gen) in data["gen"]
            gen_id = parse(Int, i)
            if abs(solution[t][gen_id] - solution[t-1][gen_id]) > ramping_data["ramp_limits"][gen_id]
                return false
            end
        end
    end
    
    # Check demands are met
    if !check_demands_met(solution, demands)
        return false
    end
    
    # Create and solve test model to verify network constraints
    try
        test_model = create_search_model(factory, 1, ramping_data, [demands[1]])
        for (i, pg) in solution[1]
            set_start_value(test_model.model[:pg][1,i], pg)
        end
        optimize!(test_model.model)
        
        if termination_status(test_model.model) != MOI.OPTIMAL && 
           termination_status(test_model.model) != MOI.LOCALLY_SOLVED
            return false
        end
    catch
        return false
    end
    
    return true
end