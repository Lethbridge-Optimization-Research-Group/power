using Graphs, MetaGraphs

function find_largest_time_period(time_periods, demands)

    largestIndex = -1
    largest = 0

    for t in 1:time_periods
        period_sum = sum(values(demands[t]))
        if period_sum > largest
            largest = period_sum
            largestIndex = t
        end
    end

    return largestIndex
end

function build_and_optimize_largest_period(factory, demand, ramping_data)

    model = create_search_model(factory, 1, ramping_data, [demand])
    optimize!(model.model)

    return model
end

function generate_random_loads(largest_model; scenarios_to_generate = 7, variation_percent = 1)
    # Used to check that conversion to Dict did not upset order
    #pg_values = value.(largest_model.model[:pg])

    # Extract largest values
    if termination_status(largest_model.model) == MOI.INFEASIBLE
        error("Largest model is infeasible. Cannot generate random loads.")
    end
    
    largest_values = [value(largest_model.model[:pg][key]) for key in keys(largest_model.model[:pg])]
    
    sum_of_largest = sum(largest_values)
    # Pair values with corresponding generator number
    pg_values = Dict(zip(largest_model.model[:pg].axes[2], largest_values))

    random_scenarios = Vector{Dict{Int64, Float64}}(undef, scenarios_to_generate)

    for t in 1:scenarios_to_generate
        random_dict = Dict()
        pos_or_neg = rand([0.35, 0.5, 0.65]) # randomly select, < will decrease, > will increase
        for (gen_num, gen_output) in pg_values
            max_variation = gen_output * (variation_percent/100)
            variation = rand() * max_variation
            if rand() >= pos_or_neg
                random_dict[gen_num] = gen_output + variation
            else
                random_dict[gen_num] = gen_output - variation
            end
        end

        random_scenarios[t] = random_dict

        variation_percent += 1
    end
    return random_scenarios
end

function generate_new_loads(current_outputs; scenarios_to_generate = 7, variation_percent = 1)
    # TODO: for the first iterations, focus primarily on decrease (variation can be larger as well)
    # TODO: change only a few generators each iteration
    # TODO: Randomly pick subset of gens, change only those
    # total_it - current_it / total_it = pos_or_neg value for first iterations
    random_scenarios = Vector{Dict{Int64, Float64}}(undef, scenarios_to_generate)
    for i in 1:scenarios_to_generate
        random_dict = Dict()
        pos_or_neg = rand([0.15, 0.5, 0.85])
        for (gen, val) in current_outputs
            max_variation = val * (variation_percent/100)
            variation = rand() * max_variation
            if rand() >= pos_or_neg
                random_dict[gen] = val + variation
            else
                random_dict[gen] = val - variation
            end
        end
        random_scenarios[i] = random_dict
        variation_percent += 1
    end
    #push!(random_scenarios, current_outputs)
    return random_scenarios
end

function power_flow(factory, demand, ramping_data, load)

    model = create_search_model(factory, 1, ramping_data, [demand])
    set_optimizer_attribute(model.model, "LogToConsole", 0)
    for (gen_id, value) in load
        fix(model.model[:pg][1,gen_id], value, force=true)
    end
    optimize!(model.model)

    return model
end

function extract_power_flow_data(model)
    
    m = value.(model.model[:pg])
    values = [value(m[key]) for key in keys(m)]
    return Dict(zip(m.axes[2], values'))
end

function test_scenarios(factory, demand, ramping_data, random_scenarios)
    feasible_scenarios = []
    minimum_demand = sum(values(demand))
    for scenario in random_scenarios
            model = power_flow(factory, demand, ramping_data, scenario)
            status = termination_status(model.model)
            if status != MOI.LOCALLY_SOLVED && status != MOI.OPTIMAL
                println("Skipping infeasible scenario")
                continue  # Skip extracting values from an infeasible model
            end
            if (sum(value.(model.model[:pg])) < minimum_demand)
                println("Demand not met, skipping scenario")
                continue # Skip if demand is not met
            end
            values = extract_power_flow_data(model)
            push!(feasible_scenarios, (values, objective_value(model.model)))
    end
    return feasible_scenarios
end

function build_initial_graph(scenarios::Vector{Any}, time_periods)
    graph = MetaDiGraph()
    defaultweight!(graph, 1.0)
    
    # add first node
    add_vertex!(graph)
    first_node = nv(graph)
    set_prop!(graph, first_node, :time_period, 0)
    set_prop!(graph, first_node, :generator_values, 0)
    set_prop!(graph, first_node, :cost, 0)

    for p in 1:time_periods
        for (t, scenario) in enumerate(scenarios)
            add_vertex!(graph)
            current_node = nv(graph)
            
            set_prop!(graph, current_node, :time_period, p) # time period
            set_prop!(graph, current_node, :generator_values, scenario[1]) # generator values
            set_prop!(graph, current_node, :cost, scenario[2]) # sum of values
        end
    end

    # add last node and corresponding edges
    add_vertex!(graph)
    last_node = nv(graph)
    set_prop!(graph, last_node, :time_period, time_periods + 1)
    set_prop!(graph, last_node, :generator_values, 0)
    set_prop!(graph, last_node, :cost, 0)

    first_nodes = collect(filter_vertices(graph, :time_period, 1))
    for n in first_nodes
        add_edge!(graph, first_node, n)
        edge = Edge(first_node, n)
        set_prop!(graph, edge, :weight, 0)
    end

    last_nodes = collect(filter_vertices(graph, :time_period, time_periods))
    for n in last_nodes
        add_edge!(graph, n, last_node)
        edge = Edge(n, last_node)
        set_prop!(graph, edge, :weight, 0)
    end

    return graph
end

function add_weighted_edges(graph, time_periods, ramping_data)

    ramp_costs = ramping_data["costs"]
    ramp_limits = ramping_data["ramp_limits"]
    for n in 1:(time_periods - 1)
        nodes_n = collect(filter_vertices(graph, :time_period, n))
        nodes_n1 = collect(filter_vertices(graph, :time_period, n + 1))

        for node_n in nodes_n
            gen_values_n = get_prop(graph, node_n, :generator_values)
            for node_n1 in nodes_n1
                gen_values_n1 = get_prop(graph, node_n1, :generator_values)
                total_edge_cost = 0
                violates = false
                for gen_id in keys(gen_values_n)
                    difference = abs(gen_values_n[gen_id] - gen_values_n1[gen_id])
                    if difference <= ramp_limits[gen_id]
                        total_edge_cost += difference * ramp_costs[gen_id]
                    else
                        violates = true
                        break
                    end
                end
                if !violates
                    add_edge!(graph, node_n, node_n1)
                    edge = Edge(node_n, node_n1)
                    set_prop!(graph, edge, :weight, total_edge_cost)
                end
            end
        end
    end
end


function shortest_path(graph)
    # find the source node (time period 0)
    source_node = first(filter_vertices(graph, :time_period, 0))
    
    # find the sink node (time period n+1)
    sink_node = first(filter_vertices(graph, :time_period, maximum(get_prop(graph, v, :time_period) for v in vertices(graph))))
    
    # run Dijkstra's algorithm using MetaGraphs weights
    state = Graphs.dijkstra_shortest_paths(graph, source_node, MetaGraphs.weights(graph))
    
    # reconstruct path
    full_path = Int[]
    current = sink_node
    
    # start from sink  and work backward to source
    while current != source_node
        push!(full_path, current)
        current = state.parents[current]
    end
    push!(full_path, source_node)
    
    # reverse path
    reverse!(full_path)
    
    # calculate cost
    total_cost = 0.0
    for i in 1:(length(full_path)-1)
        src_node = full_path[i]
        dst_node = full_path[i+1]
        if has_edge(graph, src_node, dst_node)
            total_cost += get_prop(graph, src_node, dst_node, :weight) + get_prop(graph, src_node, :cost)
        end
    end
    
    return full_path, total_cost
end

function extract_solution(graph, path)
    solution = Dict{Int, Dict{Symbol, Any}}()  # Dictionary to store node properties

    for node in path
        time_period = get_prop(graph, node, :time_period)
        solution[time_period] = Dict(
            :generator_values => get_prop(graph, node, :generator_values),
            :cost => get_prop(graph, node, :cost)
        )
    end

    return solution
end

function build_new_graph(new_scenarios, time_periods) 

    new_graph = MetaDiGraph()
    defaultweight!(new_graph, 1.0)

    add_vertex!(new_graph)
    source_node = nv(new_graph)
    set_prop!(new_graph, source_node, :time_period, 0)
    set_prop!(new_graph, source_node, :generator_values, 0)
    set_prop!(new_graph, source_node, :cost, 0)

    for t in 1:time_periods
        for (s, scenario) in enumerate(new_scenarios[t])
            add_vertex!(new_graph)
            current_node = nv(new_graph)
            set_prop!(new_graph, current_node, :time_period, t)
            set_prop!(new_graph, current_node, :generator_values, scenario[1])
            set_prop!(new_graph, current_node, :cost, scenario[2])
        end
    end

    add_vertex!(new_graph)
    sink_node = nv(new_graph)
    set_prop!(new_graph, sink_node, :time_period, time_periods + 1)
    set_prop!(new_graph, sink_node, :generator_values, 0)
    set_prop!(new_graph, sink_node, :cost, 0)

    # Connect source to first time period nodes
    first_nodes = collect(filter_vertices(new_graph, :time_period, 1))
    for n in first_nodes
        add_edge!(new_graph, source_node, n)
        edge = Edge(source_node, n)
        set_prop!(new_graph, edge, :weight, 0)
    end
    
    # Connect last time period nodes to sink
    last_nodes = collect(filter_vertices(new_graph, :time_period, time_periods))
    for n in last_nodes
        add_edge!(new_graph, n, sink_node)
        edge = Edge(n, sink_node)
        set_prop!(new_graph, edge, :weight, 0)
    end

    return new_graph
end


function iter_search(factory, demands, ramping_data, time_periods)

    highest_demand = find_largest_time_period(time_periods, demands)
    largest_model = build_and_optimize_largest_period(factory, demands[highest_demand], ramping_data)

    loads = generate_random_loads(largest_model)

    scenarios = test_scenarios(factory, demands[highest_demand], ramping_data, loads)

    graph = build_initial_graph(scenarios, time_periods)
    add_weighted_edges(graph, time_periods, ramping_data)

    path, cost = shortest_path(graph)
    
    best_graph = graph
    best_path = path[2:end - 1]
    best_cost = cost
    best_solution = extract_solution(best_graph, best_path)

    cost_history = Vector{Float64}()
    push!(cost_history, best_cost)

    iteration = 1
    converged = false
    max_iterations = 10
    convergence_threshold = 0.01

    while !converged && iteration < max_iterations
        generator_values = Vector{Dict{Int64, Float64}}()
        for i in 1:time_periods
            time_period_vals = Dict()
            for (gen, val) in best_solution[i][:generator_values]
                time_period_vals[gen] = val
            end
            push!(generator_values, time_period_vals)
        end
        # maybe combine these two loops
        test_values = Vector{Vector{Dict{Int64, Float64}}}()
        new_feasible_values = Vector{Vector{Any}}()
        for i in 1:time_periods
            push!(new_feasible_values, Vector{Any}())
        end

        for i in 1:time_periods
            loads_for_period = generate_new_loads(generator_values[i])
            tested_loads = test_scenarios(factory, demands[i], ramping_data, loads_for_period)
            new_feasible_values[i] =  tested_loads
        end
        # new_feasible_values[1][1][1] = value Dict, [1][1][2] = total gen cost
        # run shortest path and compare values
        new_graph = build_new_graph(new_feasible_values, time_periods)
        add_weighted_edges(new_graph, time_periods, ramping_data)

        new_path, new_cost = shortest_path(new_graph)
        #=
        if abs(new_cost - best_cost) < 0.01
            println("Converged on $best_cost")
            converged = true
        end
        =#
        if new_cost < best_cost
            best_graph = new_graph
            best_cost = new_cost
            best_path = new_path
            best_solution = extract_solution(best_graph, best_path)
        end
        push!(cost_history, best_cost)
        iteration += 1
        println("Iteration: ", iteration)
    end


    display(cost_history)
    return best_graph, best_path, best_cost, best_solution, cost_history

end


function find_infeasible_constraints(model::Model)
    #if termination_status(model) != MOI.LOCALLY_INFEASIBLE
     #   println("The model must be optimized and locally infeasible")
	#	return []
    #end

    infeasible_constraints = []
    
    for (f, s) in list_of_constraint_types(model)
        for con in all_constraints(model, f, s)
            func = constraint_object(con).func
            set = constraint_object(con).set
            constraint_value = JuMP.value(func)
            
            is_satisfied = false
            if set isa MOI.EqualTo
                is_satisfied = isapprox(constraint_value, MOI.constant(set), atol=1e-6)
            elseif set isa MOI.LessThan
                is_satisfied = constraint_value <= MOI.constant(set) + 1e-6
            elseif set isa MOI.GreaterThan
                is_satisfied = constraint_value >= MOI.constant(set) - 1e-6
            elseif set isa MOI.Interval
                is_satisfied = MOI.lower(set) - 1e-6 <= constraint_value <= MOI.upper(set) + 1e-6
            else
                @warn "Unsupported constraint type: $set"
                continue
            end
            
            if !is_satisfied
                push!(infeasible_constraints, (con, constraint_value))
            end
        end
    end

    return infeasible_constraints
end

function find_bound_violations(model::Model)

	# if termination_status(model) != MOI.LOCALLY_INFEASIBLE
    #     error("The model must be optimized and locally infeasible")
    # end

	# Get the variable names
	variable_names = all_variables(model)

	violations = Dict{VariableRef, Tuple{Float64, Float64, Float64, Float64}}()

	# iterate over all variables
	for (_, var) in enumerate(variable_names)

		# check if the variable has a lower and upper bound
		if !has_lower_bound(var) || !has_upper_bound(var)
			continue
		end

		# get the bounds and value of the variable
		upper = upper_bound(var)
		lower = lower_bound(var)
		value = JuMP.value(var)

		# check for violation
		if value < lower 
			
			# add it to the violations dictionary
			violations[var] = (value, lower, upper, lower - value)
		elseif value > upper

			# add it to the violations dictionary
			violations[var] = (value, lower, upper, value - upper)
		end
	end
    	# return the violations
	return violations
end

function print_path_details(graph, path)
    println("Path Details:")
    println("------------")
    
    # Iterate through each node in the path
    for i in 1:length(path)
        node = path[i]
        
        # Print node information
        println("Node $i: $node")
        println("  Time Period: ", get_prop(graph, node, :time_period))
        println("  Generator Values: ", get_prop(graph, node, :generator_values))
        println("  Cost: ", get_prop(graph, node, :cost))
        
        # Print edge information (if not the last node)
        if i < length(path)
            next_node = path[i+1]
            if has_edge(graph, node, next_node)
                edge_weight = get_prop(graph, node, next_node, :weight)
                println("  Edge to $(next_node):")
                println("    Weight: ", edge_weight)
            else
                println("  No edge exists to $(next_node)")
            end
        end
        println()
    end
    
    # Calculate and print total path cost
    total_cost = sum(get_prop(graph, node, :cost) for node in path)
    total_edge_weight = 0.0
    for i in 1:(length(path)-1)
        if has_edge(graph, path[i], path[i+1])
            total_edge_weight += get_prop(graph, path[i], path[i+1], :weight)
        end
    end
    println("Total Path Cost: ", total_cost + total_edge_weight)
end