using JuMP, PowerModels, Ipopt, Random
using MPOPF
#=
Idea: Move from highest to lowest demand time periods, optimize the time period and set adjacent time 
periods to the current time periods generator values - each generators respective ramping limits if these adjacent
time periods are currently set to produce LESS than the current time period. If we encounter a time period that has
already been set (we've optimized the time period before or after and set this one to that time periods output - ramping limits)
only update the values if they are GREATER than what is already present. (This way we maintain feasibility and 
ramping constraints arent violated)

There exist edge cases that check_ramping_constraints and apply_ramping_constraints should catch 
=#
function create_initial_solution(factory, data, time_periods, demands, ramping_data)
    solution = []
    models = []

    # Set all generators to inital value of 0
    for t in 1:time_periods
        pg = Dict()
        for (i, gen) in data["gen"]
            gen_id = parse(Int, i)
            pg[gen_id] = 0
        end
        push!(solution, pg)
    end

    # Sort time periods by total demand in non-decreasing order
    sorted_demands = deepcopy(demands)
    sorted_demands = sorted_demands[1:time_periods]
    sorted_demands = collect(enumerate(sorted_demands))
    sorted_demands = sort(sorted_demands, by = x-> sum(x[2]), rev = true)
    
    for t in 1:time_periods
        current_t = sorted_demands[t][1]
        model = create_search_model(factory, 1, ramping_data, [sorted_demands[t][2]])
        optimize_model(model)

        gen_indices = axes(model.model[:pg], 2)

        pg_result = Dict{Int, Float64}()
        for (i, gen_id) in enumerate(gen_indices)
            pg_result[gen_id] = value(model.model[:pg][1, gen_id])
            pg_result[gen_id] = max(pg_result[gen_id], data["gen"][string(gen_id)]["pmin"], 0)
        end
        
        if sum(values(pg_result)) > sum(values(solution[current_t]))
            solution[current_t] = pg_result
        end
#=
        if current_t > 1 # There is a time period before that needs to be adjusted
            if sum(values(solution[current_t])) > sum(values(solution[current_t - 1]))
                adjust_adjacent_time_period!(current_t, solution, ramping_data, :before)
            end
        end
        if current_t < time_periods # There is a time period after that needs to be adjusted
            if sum(values(solution[current_t])) > sum(values(solution[current_t + 1]))
                adjust_adjacent_time_period!(current_t, solution, ramping_data, :after)
            end
        end
        =#
        push!(models,(current_t,model))
    end

    no_ramping_violations = check_ramping_constraints(solution, ramping_data)

    while no_ramping_violations == false
        apply_ramping_constraints(solution, ramping_data)
        no_ramping_violations = check_ramping_constraints(solution, ramping_data)
    end

    return solution, models
end


function adjust_adjacent_time_period!(current_t, solution, ramping_data, direction)
    # Set generators in adjacent (before or after) time periods to current time period values - ramping limits
    # or to their minimum outputs, whichever is larger
    for (gen_id, value) in solution[current_t]
        if direction == :before
            adjacent_t = current_t - 1
            solution[adjacent_t][gen_id] = value - ramping_data["ramp_limits"][gen_id]
            solution[adjacent_t][gen_id] = max(solution[adjacent_t][gen_id], data["gen"][string(gen_id)]["pmin"])
        else # :after
            adjacent_t = current_t + 1
            solution[adjacent_t][gen_id] = value - ramping_data["ramp_limits"][gen_id]
            solution[adjacent_t][gen_id] = max(solution[adjacent_t][gen_id], data["gen"][string(gen_id)]["pmin"])
        end
    end
end

function check_ramping_constraints(solution, ramping_data) 
    time_periods = length(solution)
    for t in 2:time_periods
        for (gen_id, current_output) in solution[t]
            prev_output = solution[t-1][gen_id]
            ramp_limit = ramping_data["ramp_limits"][gen_id]

            ramp_up = current_output - prev_output
            ramp_down = prev_output - current_output

            if ramp_up > ramp_limit + 1e-6 # Add small tolerance for floating point errors
                return false
            elseif ramp_down > ramp_limit + 1e-6
                return false
            end
        end
    end
    return true
end

function apply_ramping_constraints(solution, ramping_data)
    time_periods = length(solution)
    violations = []

    for t in 2:time_periods
        for (gen_id, current_output) in solution[t]
            prev_output = solution[t-1][gen_id]
            ramp_limit = ramping_data["ramp_limits"][gen_id]

            ramp_up = current_output - prev_output
            ramp_down = prev_output - current_output

            if ramp_up > ramp_limit + 1e-6 # Add small tolerance for floating point errors
                push!(violations, (t, gen_id, :up))
            elseif ramp_down > ramp_limit + 1e-6
                push!(violations, (t, gen_id, :down))
            end
        end
    end

    for (t, gen_id, direction) in violations
        if direction == :up
            solution[t][gen_id] = max(solution[t - 1][gen_id] - ramping_data["ramp_limits"][gen_id], 0)
        else 
            solution[t][gen_id] = solution[t - 1][gen_id] + ramping_data["ramp_limits"][gen_id]
        end
    end
    return solution
end

function verify_minimum_demands(solution, demands)
    all_demands_met = true
    time_periods = length(solution)
    for (t, total) in enumerate(solution)
        output = sum(values(solution[t]))
        demand = sum(demands[t])

        if output < demand - 1e-4
            println("Time period $t violation")
            println("Actual output: $output. Expected $demand, difference of: ", output - demand)
            all_demands_met = false
        end
    end
    println("All demands are met:")
    return all_demands_met
end