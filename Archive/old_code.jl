
function single_variable_search(power_flow_model, T, num_of_gens, epsilon)
    current_pg = value.(power_flow_model.model[:pg])
    current_obj_value = objective_value(power_flow_model.model)
    new_pg = current_pg
    new_obj_value = current_obj_value
    iterations = 0
    diff_of_solutions = Inf
    
    while iterations < 5 && diff_of_solutions > 0.01
        results = []
        for t in 1:T
            for g in 1:num_of_gens
                for i in 1:2
                    temp = single_variable_solve(power_flow_model, epsilon, t, g)
                    push!(results, temp)
                    # Re-optimize model to avoid querying results on modified model
                    optimize!(power_flow_model.model)
                    epsilon *= -1
                end
            end
        end
        # Filter solved models for cheapest objective value
        solved_solutions = filter(x -> x[3] == "LOCALLY_SOLVED", results)
        min_solution = solved_solutions[argmin(x[1] for x in solved_solutions)]
        # If cheapest found value is better than our current,
        # set update variables and reset loop with new variables
        if min_solution[1] < new_obj_value
            new_obj_value = min_solution[1]
            new_pg = min_solution[2]
            fix.(power_flow_model.model[:pg], new_pg; force=true)
            iterations = 0
            optimize!(power_flow_model.model)
        end
        # Calculate difference in solutions, if incremental, terminates loop
        diff_of_solutions = abs(new_obj_value - current_obj_value)
        iterations += 1
    end
    # Return the cheapest value with its corresponding pg variables
    return new_obj_value, new_pg
end

function single_variable_solve(power_flow_model, epsilon, i, j)
    # Store current values before modifying the model
    current_pg = power_flow_model.model[:pg]
    changed_pg = value(current_pg[i, j]) + epsilon
    must_unfix = false
    
    # Save current upper and lower bounds of the variable
    upper = power_flow_model.data["gen"][string(j)]["pmax"]
    lower = power_flow_model.data["gen"][string(j)]["pmin"]
    
    # Fix variable to the modified value
    if changed_pg >= lower && changed_pg <= upper
        fix(current_pg[i, j], changed_pg; force = true)
        must_unfix = true
    end

    # Optimize the model
    optimize!(power_flow_model.model)

    status = termination_status(power_flow_model.model)
    status = string(status)
    obj_value = objective_value(power_flow_model.model)
    new_pg = value.(current_pg)

    # Unfix the variable and restore the upper and lower bounds
    if must_unfix == true
        unfix(current_pg[i,j])
        set_upper_bound(current_pg[i,j], upper)
        set_lower_bound(current_pg[i,j], lower)
    end
    return obj_value, new_pg, status
end

function two_variable_search(power_flow_model, T, num_of_gens, epsilon)
    current_pg = value.(power_flow_model.model[:pg])
    current_obj_value = objective_value(power_flow_model.model)
    new_pg = current_pg
    new_obj_value = current_obj_value
    iterations = 0
    diff_of_solutions = Inf

    while iterations < 5 && diff_of_solutions > 0.01
        results = []
        for i in 1:T
            for j in 1:num_of_gens - 1
                for k in 1:2
                    temp = two_variable_solve(power_flow_model, epsilon, i, j, j + 1)
                    push!(results, temp)
                    optimize!(power_flow_model.model)
                    if temp[1] < new_obj_value
                        new_obj_value = temp[1]
                        new_pg = temp[2]
                        fix.(power_flow_model.model[:pg], temp[2], force=true)
                        optimize!(power_flow_model.model)
                    end
                    epsilon *= -1
                end
            end
        end
    
        solved_solutions = filter(x -> x[3] == "LOCALLY_SOLVED", results)
        min_solution = solved_solutions[argmin(x[1] for x in solved_solutions)]

        if min_solution[1] < new_obj_value
            new_obj_value = min_solution[1]
            new_pg = min_solution[2]
            fix.(power_flow_model.model[:pg], new_pg; force=true)
            iterations = 0
            optimize!(power_flow_model.model)
        end
        diff_of_solutions = abs(new_obj_value - current_obj_value)
        iterations += 1
    end
    return new_obj_value, new_pg
end

function two_variable_solve(power_flow_model, epsilon, i, j, k)
    current_pg = power_flow_model.model[:pg]
    changed_pg1 = value(current_pg[i,j]) + epsilon
    changed_pg2 = value(current_pg[i,k]) + epsilon
    must_unfix = false

    upper1 = power_flow_model.data["gen"][string(j)]["pmax"]
    lower1 = power_flow_model.data["gen"][string(j)]["pmin"]
    upper2 = power_flow_model.data["gen"][string(k)]["pmax"]
    lower2 = power_flow_model.data["gen"][string(k)]["pmin"]
    
    if (changed_pg1 >= lower1 && changed_pg1 <= upper1) &&
       (changed_pg2 >= lower2 && changed_pg2 <= upper2)
        fix(current_pg[i,j], changed_pg1; force = true)
        fix(current_pg[i, k], changed_pg2; force = true)
        must_unfix = true
    end

    optimize!(power_flow_model.model)

    status = termination_status(power_flow_model.model)
    status = string(status)
    obj_value = objective_value(power_flow_model.model)
    new_pg = value.(current_pg)

    if must_unfix == true
        unfix(current_pg[i,j])
        unfix(current_pg[i, k])
        set_upper_bound(current_pg[i,j], upper1)
        set_lower_bound(current_pg[i,j], lower1)
        set_upper_bound(current_pg[i, k], upper2)
        set_lower_bound(current_pg[i, k], lower2)
    end

    return obj_value, new_pg, status
end


### Not finished, do not call
function single_nonmonotone_search(power_flow_model, T, num_of_gens, epsilon)
    current_pg = value.(power_flow_model.model[:pg])
    current_obj_value = objective_value(power_flow_model.model)
    new_pg = current_pg
    new_obj_value = current_obj_value

    previous_values = [Inf, Inf, Inf, Inf, Inf]

    while new_obj_value < maximum(map(x -> x[1], previous_values))
        results = []
        for i in 1:40
            a = rand(1:num_of_gens)
            b = rand(1:num_of_gens)
            t = rand(1:T)
            while a == b
                b = rand(1:num_of_gens)
            end
            temp = two_variable_solve(power_flow_model, epsilon, t, a, b)
            push!(results, temp)
            optimize!(power_flow_model.model)
        end

        solved_solutions = filter(x -> x[3] == "LOCALLY_SOLVED", results)
        min_solution = solved_solutions[argmin(x[1] for x in solved_solutions)]

        pop!(previous_values)
        pushfirst!(previous_values, min_solution[1])

        if min_solution[1] < new_obj_value
            new_obj_value = min_solution[1]
            new_pg = min_solution[2]
            fix.(power_flow_model.model[:pg], new_pg; force=true)
            optimize!(power_flow_model.model)
        end
    end
    return new_obj_value, new_pg
end



###################################################
# Variable search without using MPOPF solver 
###################################################
# 0. Optimize time periods individually, sum them and 
#    add ramping as a base cost 
# 1. Modify variables (total demand or individual generators)
# 2. Re-optimize that time period
# 3. Re-sum total cost 
# 4. Compare with solver



function single_variable_search_DC(factory::MPOPF.AbstractMPOPFModelFactory, file_path::String, T::Int, factors::Vector{Float64}, ramping_cost::Int)
    data = PowerModels.parse_file(file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    data_copy = deepcopy(data)

    base_values = calculate_base_cost(factory, data_copy, T, factors, ramping_cost)
    initial_cost = base_values[1]
    initial_pg_values = base_values[2]
    initial_data = base_values[3]
    
    best_cost = initial_cost
    best_pg = initial_pg_values
    search_data = deepcopy(initial_data)
    epsilon = 0.1

    test_vec = []
    while epsilon > 0.00001
        results = single_variable_neighbourhood_search(factory, search_data, T, factors, ramping_cost, epsilon)
        push!(test_vec, results[1])
        if results[1] < best_cost
            best_cost = results[1]
            best_pg = results[2]
            search_data = deepcopy(results[3])
        else
            epsilon /= 10
        end
        # Need to add a vector of data probably
        # Store data by index of T so that consecutive runs 
        # are cumulative
    end

    return initial_cost, best_cost, best_pg, test_vec
end

function calculate_base_cost(factory::MPOPF.AbstractMPOPFModelFactory, data, T, factors, ramping_cost)
    cost_per_time_period = []
    pg_values_per_time_period = []
    statuses_per_time_period = []
    data_per_time_period  = []
    for t in 1:T
        data_copy = deepcopy(data)
        for (bus_id, load) in data_copy["load"]
            data_copy["load"][bus_id]["pd"] *= factors[t]
        end
        new_model = MPOPF.create_search_model(factory, data_copy)
        optimize!(new_model.model)

        status = termination_status(new_model.model)
        status = string(status)

        push!(cost_per_time_period, objective_value(new_model.model))
        push!(pg_values_per_time_period, value.(new_model.model[:pg]))
        push!(statuses_per_time_period, status)
        push!(data_per_time_period, data_copy)
    end

    if any(status != "LOCALLY_SOLVED" for status in statuses_per_time_period)
        error("Optimization did not converge locally for all time periods. One or more time periods may be infeasible. Change in demand may exceed constraints.")
    end
    
    total_ramping = sum(abs(cost_per_time_period[i] - cost_per_time_period[i-1]) for i in 2:T)
    base_cost = sum(cost_per_time_period) + total_ramping * ramping_cost

    return base_cost, pg_values_per_time_period, data_per_time_period
end

function single_variable_neighbourhood_search(factory::MPOPF.AbstractMPOPFModelFactory, data, T, factors, ramping_cost, epsilon)
    results = []
    for i in 1:T
        data_vec = []
        costs = []
        pg_values = []
        statuses = []
        for t in 1:T
            data_copy = deepcopy(data[t])
            if i == t
                for (bus_id, load) in data_copy["load"]
                    data_copy["load"][bus_id]["pd"] *= (factors[t] + epsilon)
                end
            else
                for (bus_id, load) in data_copy[t]["load"]
                    data_copy["load"][bus_id]["pd"] *= (factors[t])
                end
            end
            new_model = MPOPF.create_search_model(factory, data_copy[t])
            optimize!(new_model.model)

            status = termination_status(new_model.model)
            status = string(status)

            push!(costs, objective_value(new_model.model))
            push!(pg_values, value.(new_model.model[:pg]))
            push!(statuses, status)
            push!(data_vec, data_copy)
        end

        if any(status != "LOCALLY_SOLVED" for status in statuses)
           total_cost = Inf
        else
            iteration_cost = sum(costs)
            ramping = sum(abs(costs[i] - costs[i-1]) for i in 2:T)
            total_cost = iteration_cost + ramping * ramping_cost
        end
        push!(results, (total_cost, pg_values, deepcopy(data_copy)))
    end
    
    return results[argmin(x[1] for x in results)]
end

function search_gens(factory, file_path, T, factors, ramping_cost)
    # Array of models solved individually, can modify and re-optimize
    # individual models and recalculate total costs

end




#= 
Implement some sort of "slack" constraint, this allows for +/- overall demand
for some time period and re-evaluate the rest of the model and see if it is cheaper
Keep in mind constraints on ramp up/ramp down between time periods. T2s value must be feasible
in respect to the amount of ramping each gen can do compared to T1

* some max value for ramping where
Pt - Pt-1 <= max
Pt-1 - Pt <= max

Start from the highest demand time period and build models out from there
1. Apply individual costs to generators
2. Constraints of generator ramping are some % of their total output
(For now at least, this may change but this gets us a model)

Hard code values for gen ramping limits (vector)
Hard code values for gen ramping costs (unique per generator)
Add a slack constraint (slack can initialize as 0 at first) and pick some slack
such that we can change demand by some epsilon to converge on some optimal solution
=#
# Look into presentation mode / MARP
