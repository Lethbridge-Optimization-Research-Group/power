using Graphs, MetaGraphs

function DC_graph_search(data, factory, demands, ramping_data, time_periods)

    total_violations = 0

    highest_demand = find_largest_time_period(time_periods, demands)
    largest_model = build_and_optimize_largest_period(factory, demands[highest_demand], ramping_data)

    loads = generate_random_loads(largest_model)

    scenarios = test_scenarios(data, factory, demands[highest_demand], ramping_data, loads)

    graph = build_initial_graph(scenarios, time_periods)
    add_weighted_edges!(graph, time_periods, ramping_data)

    feasibility = false
    path = 0.0
    while !feasibility

        path = shortest_path(graph, time_periods)

        infeasible_nodes = test_feasibility(factory, path, graph, demands, ramping_data)

        if isempty(infeasible_nodes) 
            feasibility = true
        else
            # Remove infeasible nodes and rebuild connections
            for node in sort(infeasible_nodes, rev=true)  # Remove in reverse order to maintain indices
                rem_vertex!(graph, node)
            end
            # Need to rebuild edges after removing vertices
            add_weighted_edges!(graph, time_periods, ramping_data)
        end
    end

    path_results = calculate_path_cost(path, graph)
    
    best_graph = graph
    best_path = path
    best_cost = path_results[:total_cost]
    best_solution = extract_solution(best_graph, best_path)

    cost_history = Vector{Float64}()
    push!(cost_history, best_cost)

    iteration = 1
    converged = false
    max_iterations = 50
    convergence_threshold = 0.01

    generation_cost = 0.0
    ramping_cost = 0.0

    while iteration < max_iterations

        current_generator_values = Vector{Dict{Int64, Float64}}()
        new_generator_values = Vector{Vector{Any}}()

        for i in 1:time_periods
            time_period_values = Dict()
            for (gen_id, val) in best_solution[i][:generator_values]
                time_period_values[gen_id] = val
            end
            push!(current_generator_values, time_period_values)
            push!(new_generator_values, Vector{Any}())
        end

        for i in 1:time_periods
            scenarios_for_period = generate_new_scenarios(current_generator_values[i], iteration)
            tested_scenarios = test_scenarios(data, factory, demands[i], ramping_data, scenarios_for_period)
            new_generator_values[i] = tested_scenarios
        end

        new_graph = build_new_graph(new_generator_values, time_periods)
        add_weighted_edges!(new_graph, time_periods, ramping_data)

        feasibility = false

        while !feasibility

            path = shortest_path(new_graph, time_periods)

            infeasible_nodes = test_feasibility(factory, path, new_graph, demands, ramping_data)

            if isempty(infeasible_nodes) 
                feasibility = true
            else
                for node in infeasible_nodes
                    rem_vertex!(new_graph, node)
                end
            end
        end
        
        path_results = calculate_path_cost(path, new_graph)

        if path_results[:total_cost] < best_cost
            best_graph = new_graph
            best_cost = path_results[:total_cost]
            best_path = path
            best_solution = extract_solution(best_graph, best_path)
            generation_cost = path_results[:generation_cost]
            ramping_cost = path_results[:ramping_cost]
        end

        push!(cost_history, best_cost)
        iteration += 1
    end

    info = Dict(
        :graph => best_graph,
        :path => best_path,
        :cost => best_cost,
        :solution => best_solution,
        :cost_history => cost_history,
        :total_violations => total_violations,
        :generation_cost => generation_cost,
        :ramping_cost => ramping_cost
    )

    return info
end


function shortest_path(graph, time_periods)

    working_graph = deepcopy(graph)

    for e in edges(working_graph)
        src = Graphs.src(e)
        dst = Graphs.dst(e)
        current_weight = get_prop(working_graph, src, dst, :weight)
        node_cost = get_prop(working_graph, src, :cost)
        set_prop!(working_graph, src, dst, :weight, current_weight + node_cost)
    end

    # find the source node and sink nodes
    source_node = 1
    sink_node = first(filter_vertices(working_graph, :time_period, time_periods + 1))

    state = Graphs.dijkstra_shortest_paths(working_graph, source_node, MetaGraphs.weights(working_graph))

    if state.parents[sink_node] == 0 && sink_node != source_node
        return false
    end

    path = Int[]
    current = sink_node

    while current != source_node
        push!(path, current)
        current = state.parents[current]
    end
    push!(path, source_node)

    reverse!(path)

    return path
end

function test_feasibility(factory, path, graph, demands, ramping_data)
    
    infeasible_nodes = []
    for node in path[2:end-1]
        time_period = get_prop(graph, node, :time_period)
        generator_values = get_prop(graph, node, :generator_values)

        model = create_search_model(factory, 1, ramping_data, [demands[time_period]])
        set_optimizer_attribute(model.model, "LogToConsole", 0)

        for (gen_id, value) in generator_values
            fix(model.model[:pg][1, gen_id], value, force=true)
        end

        optimize!(model.model)
        status = termination_status(model.model)

        if status != MOI.LOCALLY_SOLVED && status != MOI.OPTIMAL
            push!(infeasible_nodes, node)
            continue
        end

        set_prop!(graph, node, :cost, objective_value(model.model))

    end

    return infeasible_nodes
end

function calculate_path_cost(path, graph) 

    total_cost = 0.0
    generation_cost = 0.0
    ramping_cost = 0.0

    for i in 1:(length(path)-1)
        src_node = path[i]
        dst_node = path[i+1]
        
        # Add generation cost from source node
        if has_prop(graph, src_node, :cost)
            node_cost = get_prop(graph, src_node, :cost)
            generation_cost += node_cost
            total_cost += node_cost
        end
        
        # Add ramping cost from edge
        if has_edge(graph, src_node, dst_node)
            edge_cost = get_prop(graph, src_node, dst_node, :weight)
            ramping_cost += edge_cost
            total_cost += edge_cost
        end
    end
    
    return Dict(
        :total_cost => total_cost,
        :generation_cost => generation_cost,
        :ramping_cost => ramping_cost
    )
end

function test_scenarios(data, factory, demand, ramping_data, random_scenarios)

    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    minimum_demand = sum(values(demand))

    tested_scenarios = []
    
    for scenario in random_scenarios
        
        # test that minimum demand is met
        if sum(values(scenario)) < minimum_demand
            println("Demand not met, skipping scenario")
            continue # Skip if demand is not met
        end

        calculated_cost = 0.0
        # test pmin and pmax values prior to making model, calculate cost
        for (gen_id, value) in scenario
            if !(data["gen"][string(gen_id)]["pmin"] - 0.001 <= value <= data["gen"][string(gen_id)]["pmax"])
                println("Pmin or Pmax bounds violated, skipping scnenario")
                continue # skip if pmin or pmax are violated
            end
            calculated_cost += gen_data[gen_id]["cost"][1]*value^2 +
                               gen_data[gen_id]["cost"][2]*value +
                               gen_data[gen_id]["cost"][3]
        end
        push!(tested_scenarios, (scenario, calculated_cost))
    end
    return tested_scenarios
end


### OLD FUNCTIONS ###
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

function generate_random_loads(largest_model; scenarios_to_generate = 15, variation_percent = 1)
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

function generate_new_scenarios(current_outputs, iteration; scenarios_to_generate = 15, variation_percent = 1)
    random_scenarios = Vector{Dict{Int64, Float64}}(undef, scenarios_to_generate)
    for i in 1:scenarios_to_generate
        random_dict = Dict()
        pos_or_neg = max(0.65, 1 - floor(iteration / 10) / 10)
        for (gen, val) in current_outputs
            max_variation = val * (variation_percent/100)
            variation = rand() * max_variation
            if rand() >= pos_or_neg
                random_dict[gen] = val + variation
            else
                random_dict[gen] = max(0, val - variation)
            end
        end
        random_scenarios[i] = random_dict
        variation_percent += 1
    end
    push!(random_scenarios, current_outputs)
    return random_scenarios
end

function extract_power_flow_data(model)
    
    m = value.(model.model[:pg])
    values = [value(m[key]) for key in keys(m)]
    return Dict(zip(m.axes[2], values'))
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

function add_weighted_edges!(graph, time_periods, ramping_data)

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

