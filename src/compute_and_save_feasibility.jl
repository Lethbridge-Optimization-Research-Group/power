"""
    compute_and_save_feasibility(factory::Union{ACMPOPFModelFactory, DCMPOPFModelFactory, LinMPOPFModelFactory}, path::String, model_type=false)

Compute the feasibility of a model and save the results to a specified path.

# Arguments
- `factory::Union{ACMPOPFModelFactory, DCMPOPFModelFactory, LinMPOPFModelFactory}`: The factory to create the model.
- `path::String`: The path where the results will be saved.
- `model_type`: The type of the model (optional, only needed for `LinMPOPFModelFactory`).

# Returns
- `Dict`: A dictionary containing the results of the computation.
"""
function compute_and_save_feasibility(factory::Union{ACMPOPFModelFactory, DCMPOPFModelFactory, LinMPOPFModelFactory}, path::String, model_type=false)

	# start the time
	start_time = time()

	factory_type = "error"

	if factory isa ACMPOPFModelFactory
		factory_type = "AC"

	elseif factory isa DCMPOPFModelFactory
		factory_type = "DC"

	elseif factory isa LinMPOPFModelFactory
		if model_type == Lin1
			factory_type = "Linear"

		elseif model_type == Lin2
			factory_type = "Quadratic"

		elseif model_type == Lin3
			factory_type = "Logarithmic"
		end
	end

    # Create directory structure
    case_name = replace(basename(path), ".m" => "")
    base_dir = "results"
    case_dir = joinpath(base_dir, case_name)
    factory_dir = joinpath(case_dir, factory_type)
    mkpath(factory_dir)

	# initiate the data
	costs = Dict()
	v_error = Dict()
	o_error = Dict()
	times = Dict()

	# initiate and optimize the model
	model_type !== false ? model = create_model(factory; model_type=model_type) : model = create_model(factory)

	optimize_model(model)

	termination_status = JuMP.termination_status(model.model)
	statusString = string(termination_status)
	if statusString != "LOCALLY_SOLVED"
		file = joinpath(factory_dir, "status.jls")
		open(file, "w") do io
			serialize(io, termination_status)
		end
	end

	# Serialize the optimized model
	model_file = joinpath(factory_dir, "optimized_model.jls")
	open(model_file, "w") do io
		serialize(io, model)
	end

	# get the length of the buses
	ref = get_ref(model.data)
	bus_len = length(ref[:bus])

	# extract the pg and qg values
	pg = value.(model.model[:pg])
	qg = factory isa DCMPOPFModelFactory ? 0 : value.(model.model[:qg])

	# create ac model with fixed pg and qg values
	ac_factory = NewACMPOPFModelFactory(path, Ipopt.Optimizer)
	ac_model = create_model_check_feasibility(ac_factory, pg, qg)
	optimize_model(ac_model)

	termination_status_AC = JuMP.termination_status(ac_model.model)
	statusString_AC = string(termination_status_AC)
	if statusString_AC != "LOCALLY_SOLVED"
		file = joinpath(factory_dir, "status_AC.jls")
		open(file, "w") do io
			serialize(io, termination_status_AC)
		end
	end

	# get va values from the model ( it is horrible as bus indices are not always linearly increasing )
	val1 = value.(getindex.((pairs(cat(model.model[:va], dims=1)) |> collect)[1:length(model.model[:va])], 2))
	val2 = value.(getindex.((pairs(cat(ac_model.model[:va], dims=1)) |> collect)[1:length(ac_model.model[:va])], 2))
	new_o_error = abs(sum((val1 - val2) / val2) / bus_len) 

	# get vm values from the model, dc models do not have vm values, so default to 1
	# val3 = factory isa DCMPOPFModelFactory ? 0 : value.(getindex.((pairs(cat(model.model[:vm], dims=1)) |> collect)[1:length(model.model[:va])], 2))
	# val4 = factory isa DCMPOPFModelFactory ? 0 : value.(getindex.((pairs(cat(ac_model.model[:vm], dims=1)) |> collect)[1:length(ac_model.model[:va])], 2))
	# new_v_error = factory isa DCMPOPFModelFactory ? 1 : abs(sum((val3 - val4) / val4) / bus_len)

    val3 = value.(getindex.((pairs(cat(model.model[:vm], dims=1)) |> collect)[1:length(model.model[:va])], 2))
	val4 = value.(getindex.((pairs(cat(ac_model.model[:vm], dims=1)) |> collect)[1:length(ac_model.model[:va])], 2))
	new_v_error = abs(sum((val3 - val4) / val4) / bus_len)

	# calculate sum of x over sum of pg from inital model to show feasibility
	sum_x = sum(value.(ac_model.model[:x]))
	sum_pg = sum(pg)
	sum_total = sum_x / sum_pg

	# multiply value with cost
	cost_Lin = objective_value(ac_model.model)
	total_cost = sum_total * cost_Lin

	# push the calculate values
	costs[path] = total_cost
	v_error[path] = new_v_error
	o_error[path] = new_o_error
	times[path] = time() - start_time

	# output the v values from the optimized model
	vm = ac_model.model[:vm]
	for (i, branch) in ref[:branch]
		vm_fr = vm[1,branch["f_bus"]]
		vm_to = vm[1,branch["t_bus"]]

		for (vi, vj) in zip(values.(vm_fr), values.(vm_to))
			println(value(vi), " -> ", value(vj))
			output_to_file("$(path) -> $(log(value(vi)) - log(value(vj)))", "v_values/vm_diff.txt")
		end
	end

    # Create a dictionary with all the results
    results = Dict(
        "costs" => costs,
        "v_error" => v_error,
        "o_error" => o_error,
        "times" => times
    )

    # Serialize the results
    output_file = joinpath(factory_dir, "results.jls")
    open(output_file, "w") do io
        serialize(io, results)
    end

    return results
end

"""
    load_and_graph_results(results_directory::String, save_to_file::Bool=false)

Load results from a directory and generate graphs for feasibility, voltage magnitude error, voltage angle error, and computation times.

# Arguments
- `results_directory::String`: The directory where the results are stored. To get results run the function `compute_and_save_feasibility`.
- `save_to_file::Bool`: Whether to save the generated graphs to a file (optional).

# Returns
- `feasibility_graph`, `v_error_graph`, `o_error_graph`, and `time_graph`.
"""
function load_and_graph_results(results_directory::String, save_to_file::Bool=false)
    # Define model types and their corresponding colors
    model_types = ["AC", "DC", "Linear", "Logarithmic", "Quadratic"]
    style_indexes = [1, 2, 3, 4, 5]

    # Initialize graphs
    feasibility_graph = Graph("Feasibility_Graphs/feasibility.pdf")
    v_error_graph = Graph("Feasibility_Graphs/v_error.pdf")
    o_error_graph = Graph("Feasibility_Graphs/o_error.pdf")
    time_graph = Graph("Feasibility_Graphs/computation_time.pdf")

    # Get all case directories
    case_dirs = sort(filter(isdir, readdir(results_directory, join=true)))
    
    # Create a list to store case information and results
    case_data = []

    for case_dir in case_dirs
        case_name = basename(case_dir)
		bus_count = nothing
        case_results = Dict()
        optimized_models = Dict()

        for model in model_types
            results_file = joinpath(case_dir, model, "results.jls")
            model_file = joinpath(case_dir, model, "optimized_model.jls")
            if isfile(results_file)
                data = open(deserialize, results_file)
                case_results[model] = data

                # Get bus count from the first available model
                if isnothing(bus_count)
                    optimized_model_file = joinpath(case_dir, model, "optimized_model.jls")
                    if isfile(optimized_model_file)
                        optimized_model = open(deserialize, optimized_model_file)
                        ref = get_ref(optimized_model.data)
                        bus_count = length(ref[:bus])
                    end
                end
            end
            # If the model file exists, load the model
            if isfile(model_file)
                model_data = open(deserialize, model_file)
                optimized_models[model] = model_data
            end
        end

        # Only add cases where we have both results and bus count
        if !isempty(case_results) && !isnothing(bus_count) && !isempty(optimized_models)
            push!(case_data, (case_name, bus_count, case_results, optimized_models))
        end
    end

    # Sort case_data based on the number of buses
    sort!(case_data, by = x -> x[2])

    # Extract sorted case names
    case_names = [x[1] for x in case_data]

    for (i, model) in enumerate(model_types)
        costs = Float64[]
        v_errors = Float64[]
        o_errors = Float64[]
        times = Float64[]

        for (_, _, case_results, optimized_model) in case_data
            if haskey(case_results, model)
                data = case_results[model]
                MPOPF_Model = optimized_model[model]
                push!(costs, first(values(data["costs"])))
                push!(v_errors, first(values(data["v_error"])))
                push!(o_errors, first(values(data["o_error"])))
                # push!(times, first(values(data["times"])))
                push!(times, JuMP.solve_time(MPOPF_Model.model))
            else
                push!(costs, NaN)
                push!(v_errors, NaN)
                push!(o_errors, NaN)
                push!(times, NaN)
            end
        end

        # Add data to graphs
        add_scatter(feasibility_graph, case_names, costs, model, style_indexes[i])
        # Only add non-DC models to v_error_graph
		if model != "DC"
			add_scatter(v_error_graph, case_names, v_errors, model, style_indexes[i])
		end
        add_scatter(o_error_graph, case_names, o_errors, model, style_indexes[i])
        add_scatter(time_graph, case_names, times, model, style_indexes[i])
    end

    # Create plots *** Has Titles ***
    # create_plot(feasibility_graph, "Feasibility of Various Models", "Cases (sorted by number of buses)", "Cost Error")
    # create_plot(v_error_graph, "Voltage Magnitude (Vm) Error of Various Models", "Cases (sorted by number of buses)", "Magnitude Error", (0.8025, 0.98))
    # create_plot(o_error_graph, "Voltage Angle (Va) Error of Various Models", "Cases (sorted by number of buses)", "Angle Error")
    # create_plot(time_graph, "Computation Time of Various Models", "Cases (sorted by number of buses)", "Time (s)")

    # Create plots *** No Titles ***
    create_plot(feasibility_graph, "", "Cases (sorted by number of buses)", "Cost Error")
    create_plot(v_error_graph, "", "Cases (sorted by number of buses)", "Magnitude Error", (0.8025, 0.98))
    create_plot(o_error_graph, "", "Cases (sorted by number of buses)", "Angle Error")
    create_plot(time_graph, "", "Cases (sorted by number of buses)", "Time (seconds)")


    # Save graphs if save_to_file is true
    if save_to_file
        save_graph(feasibility_graph)
        save_graph(v_error_graph)
        save_graph(o_error_graph)
        save_graph(time_graph)
    end

    return feasibility_graph, v_error_graph, o_error_graph, time_graph
end

"""
    load_and_graph_results_recalculate(results_directory::String, save_to_file::Bool=false)

This function is the same as `load_and_graph_results` but it recalculates the voltage magnitude and angle errors instead fo pulling them from the results dictionary.
Load results from a directory and generate graphs for feasibility, voltage magnitude error, voltage angle error, and computation times.

# Arguments
- `results_directory::String`: The directory where the results are stored. To get results run the function `compute_and_save_feasibility`.
- `save_to_file::Bool`: Whether to save the generated graphs to a file (optional).

# Returns
- `feasibility_graph`, `v_error_graph`, `o_error_graph`, and `time_graph`.
"""
function load_and_graph_results_recalculate(results_directory::String, save_to_file::Bool=false)
    # Define model types and their corresponding colors
    model_types = ["AC", "DC", "Linear", "Logarithmic", "Quadratic"]
    style_indexes = [1, 2, 3, 4, 5]

    # Initialize graphs
    feasibility_graph = Graph("Feasibility_Graphs/feasibility.pdf")
    v_error_graph = Graph("Feasibility_Graphs/v_error.pdf")
    o_error_graph = Graph("Feasibility_Graphs/o_error.pdf")
    time_graph = Graph("Feasibility_Graphs/computation_time.pdf")

    # Get all case directories
    case_dirs = sort(filter(isdir, readdir(results_directory, join=true)))
    
    # Create a list to store case information and results
    case_data = []

    for case_dir in case_dirs
        case_name = basename(case_dir)
		bus_count = nothing
        case_results = Dict()
        optimized_models = Dict()

        ac_models = Dict()

        for model in model_types
            results_file = joinpath(case_dir, model, "results.jls")
            model_file = joinpath(case_dir, model, "optimized_model.jls")

            if isfile(results_file)
                data = open(deserialize, results_file)
                case_results[model] = data

                # Get bus count from the first available model
                if isnothing(bus_count)
                    optimized_model_file = joinpath(case_dir, model, "optimized_model.jls")
                    if isfile(optimized_model_file)
                        optimized_model = open(deserialize, optimized_model_file)
                        ref = get_ref(optimized_model.data)
                        bus_count = length(ref[:bus])
                    end
                end
            end
            # If the model file exists, load the model
            if isfile(model_file)
                model_data = open(deserialize, model_file)
                optimized_models[model] = model_data
                pg = value.(model_data.model[:pg])
                if model == "DC"
                    qg = 0
                else
                    qg = value.(model_data.model[:qg])
                end
                path = "Cases/" * case_name * ".m"
                ac_factory = NewACMPOPFModelFactory(path, Ipopt.Optimizer)
                ac_model = create_model_check_feasibility(ac_factory, pg, qg)
                optimize_model(ac_model)
                ac_models[model] = ac_model

            end
        end

        # Only add cases where we have both results and bus count
        if !isempty(case_results) && !isnothing(bus_count) && !isempty(optimized_models) && !isnothing(ac_models)
            push!(case_data, (case_name, bus_count, case_results, optimized_models, ac_models))
        end
    end

    # Sort case_data based on the number of buses
    sort!(case_data, by = x -> x[2])

    # Extract sorted case names
    case_names = [x[1] for x in case_data]

    compiled_results = Dict{String, Dict{String, Dict{String, Float64}}}()

    for (i, model) in enumerate(model_types)
        costs = Float64[]
        v_errors = Float64[]
        o_errors = Float64[]
        times = Float64[]

        for (case_name, bus_len, case_results, optimized_model, ac_model) in case_data
            if haskey(case_results, model)
                data = case_results[model]
                MPOPF_Model = optimized_model[model]
                push!(costs, first(values(data["costs"])))
                new_v_error = NaN
                new_o_error = NaN
                # push!(v_errors, first(values(data["v_error"])))
                # push!(o_errors, first(values(data["o_error"])))
                try
                    if (model == "DC")
                        vm_ones = ones(length(MPOPF_Model.model[:va]))
                        val3 = value.(getindex.((pairs(cat(vm_ones, dims=1)) |> collect)[1:length(MPOPF_Model.model[:va])], 2))
                    else 
                        val3 = value.(getindex.((pairs(cat(MPOPF_Model.model[:vm], dims=1)) |> collect)[1:length(MPOPF_Model.model[:va])], 2))
                    end
                    val4 = value.(getindex.((pairs(cat(ac_model[model].model[:vm], dims=1)) |> collect)[1:length(ac_model[model].model[:va])], 2))
                    new_v_error = abs(sum((val3 - val4) / val4) / bus_len)
                    push!(v_errors, new_v_error)
                catch e
                    error("Error calculating v_error for $model in case $case_name: $e")
                    push!(v_errors, NaN)
                end

                # Calculate new o_error using the formula with :va
                try
                    val1 = value.(getindex.((pairs(cat(MPOPF_Model.model[:va], dims=1)) |> collect)[1:length(MPOPF_Model.model[:va])], 2))
                    val2 = value.(getindex.((pairs(cat(ac_model[model].model[:va], dims=1)) |> collect)[1:length(ac_model[model].model[:va])], 2))
                    new_o_error = abs(sum((val1 - val2) / val2) / bus_len)
                    push!(o_errors, new_o_error)
                catch e
                    error("Error calculating o_error for $model in case $case_name: $e")
                    push!(o_errors, NaN)
                end

                # push!(times, first(values(data["times"])))
                push!(times, JuMP.solve_time(MPOPF_Model.model))

                compiled_results[case_name] = get!(compiled_results, case_name, Dict{String, Dict{String, Float64}}())
                compiled_results[case_name][model] = get!(compiled_results[case_name], model, Dict{String, Float64}())
                compiled_results[case_name][model]["cost"] = first(values(data["costs"]))
                compiled_results[case_name][model]["v_error"] = new_v_error
                compiled_results[case_name][model]["o_error"] = new_o_error
                compiled_results[case_name][model]["time"] = JuMP.solve_time(MPOPF_Model.model)

            else
                push!(costs, NaN)
                push!(v_errors, NaN)
                push!(o_errors, NaN)
                push!(times, NaN)
            end


        end

        # Add data to graphs
        add_scatter(feasibility_graph, case_names, costs, model, style_indexes[i])
        # Only add non-DC models to v_error_graph
		# if model != "DC"
			add_scatter(v_error_graph, case_names, v_errors, model, style_indexes[i])
		# end
        add_scatter(o_error_graph, case_names, o_errors, model, style_indexes[i])
        add_scatter(time_graph, case_names, times, model, style_indexes[i])
    end

    # Create plots *** Has Titles ***
    # create_plot(feasibility_graph, "Feasibility of Various Models", "Cases (sorted by number of buses)", "Cost Error")
    # create_plot(v_error_graph, "Voltage Magnitude (Vm) Error of Various Models", "Cases (sorted by number of buses)", "Magnitude Error", (0.8025, 0.98))
    # create_plot(o_error_graph, "Voltage Angle (Va) Error of Various Models", "Cases (sorted by number of buses)", "Angle Error")
    # create_plot(time_graph, "Computation Time of Various Models", "Cases (sorted by number of buses)", "Time (s)")

    # Create plots *** No Titles ***
    create_plot(feasibility_graph, "", "Cases (sorted by number of buses)", "Cost Error")
    create_plot(v_error_graph, "", "Cases (sorted by number of buses)", "Magnitude Error", (0.8025, 0.98))
    create_plot(o_error_graph, "", "Cases (sorted by number of buses)", "Angle Error")
    create_plot(time_graph, "", "Cases (sorted by number of buses)", "Time (seconds)")


    # Save graphs if save_to_file is true
    if save_to_file
        save_graph(feasibility_graph)
        save_graph(v_error_graph)
        save_graph(o_error_graph)
        save_graph(time_graph)
    end

    return feasibility_graph, v_error_graph, o_error_graph, time_graph, compiled_results
end

"""
    load_and_compile_results(results_directory::String, save_to_file::Bool=false)

Load results from a directory and compile them into a dictionary for further analysis.

# Arguments
- `results_directory::String`: The directory where the results are stored. To get results run the function `compute_and_save_feasibility`.
- `save_to_file::Bool`: Whether to save the generated results to a file (optional) (Not yet implemented).

# Returns
- `Dict{String, Dict{String, Dict{String, Float64}}}`: A dictionary containing the results of the computation.
"""
function load_and_compile_results(results_directory::String, save_to_file::Bool=false)
    # Define model types and their corresponding colors
    model_types = ["AC", "DC", "Linear", "Logarithmic", "Quadratic"]

    # Get all case directories
    case_dirs = sort(filter(isdir, readdir(results_directory, join=true)))
    
    # Create a list to store case information and results
    case_data = []

    for case_dir in case_dirs
        case_name = basename(case_dir)
        case_results = Dict()
        optimized_models = Dict()

        for model in model_types
            results_file = joinpath(case_dir, model, "results.jls")
            model_file = joinpath(case_dir, model, "optimized_model.jls")

            if isfile(results_file)
                data = open(deserialize, results_file)
                case_results[model] = data
            end

            # If the model file exists, load the model
            if isfile(model_file)
                model_data = open(deserialize, model_file)
                optimized_models[model] = model_data
            end
        end

        # Only add cases where we have both results and bus count
        if !isempty(case_results) && !isempty(optimized_models)
            push!(case_data, (case_name, optimized_models, case_results))
        end
    end

    # Initialize nested dictionary to store results for each model and case
    model_results = Dict{String, Dict{String, Dict{String, Float64}}}()
    for model in model_types
        model_results[model] = Dict(
            "costs" => Dict{String, Float64}(),
            "v_errors" => Dict{String, Float64}(),
            "o_errors" => Dict{String, Float64}(),
            "times" => Dict{String, Float64}()
        )
    end

    # Collect results for each model and case
    for (case_name, optimized_model, case_results) in case_data
        for model in model_types
            if haskey(case_results, model)
                data = case_results[model]
                MPOPF_Model = optimized_model[model]
                model_results[model]["costs"][case_name] = first(values(data["costs"]))
                model_results[model]["v_errors"][case_name] = first(values(data["v_error"]))
                model_results[model]["o_errors"][case_name] = first(values(data["o_error"]))
                # model_results[model]["times"][case_name] = first(values(data["times"]))
                model_results[model]["times"][case_name] = JuMP.solve_time(MPOPF_Model.model)
            else
                model_results[model]["costs"][case_name] = NaN
                model_results[model]["v_errors"][case_name] = NaN
                model_results[model]["o_errors"][case_name] = NaN
                model_results[model]["times"][case_name] = NaN
            end
        end
    end

    # TODO: Add code here to save graphs if save_to_file is true

    return model_results
end

"""
    load_and_compile_models(results_directory::String)

Load `models` (Contrary to `load_and_compile_results`) from a directory and compile them into a dictionary for further analysis.
Example, getting the pg value from case14 AC: `model_results["case14"]["AC"].model[:pg]`.

# Arguments
- `results_directory::String`: The directory where the models are stored.

# Returns
- `Dict{String, Dict{String, Dict{String, Any}}}`: A dictionary containing the data of the models.
"""
function load_and_compile_models(results_directory::String)
    # Define model types and their corresponding colors
    model_types = ["AC", "DC", "Linear", "Logarithmic", "Quadratic"]

    # Get all case directories
    case_dirs = sort(filter(isdir, readdir(results_directory, join=true)))
    
    # Create a list to store case information and results
    case_data = []

    for case_dir in case_dirs
        case_name = basename(case_dir)
        case_results = Dict()

        for model in model_types
            results_file = joinpath(case_dir, model, "optimized_model.jls")
            if isfile(results_file)
                data = open(deserialize, results_file)
                case_results[model] = data
            end
        end

        # Only add cases where we have both results and bus count
        if !isempty(case_results)
            push!(case_data, (case_name, case_results))
        end
    end

    # Initialize nested dictionary to store results for each model and case
    model_results = Dict{String, Dict{String, Any}}()
    for (case_name, _) in case_data
        model_results[case_name] = Dict()
    end

    # Collect results for each model and case
    for (case_name, case_results) in case_data
        for model in model_types
            if haskey(case_results, model)
                data = case_results[model]
                model_results[case_name][model] = data
            else
                model_results[case_name][model] = NaN
            end
        end
    end

    # TODO: Add code here to save graphs if save_to_file is true

    return model_results
end

"""
    calculate_model_averages(model_results::Dict{String, Dict{String, Dict{String, Float64}}})

Calculate the average values for cost, v_error, o_error and times for all models in the dictionary

# Arguments
- `model_results::Dict{String, Dict{String, Dict{String, Float64}}}`: The dictionary of models to be analyzed.
- This `model_results` can be obtained by running the function `load_and_compile_results`.

# Returns
- `Dict{String, Dict{String, Float64}}`: A dictionary that maps the model to a metric to a value
"""
function calculate_model_averages(model_results::Dict{String, Dict{String, Dict{String, Float64}}})
    # Initialize a dictionary to store the averages for each model
    model_averages = Dict{String, Dict{String, Float64}}()

    # List of metrics we're interested in
    metrics = ["costs", "v_errors", "o_errors", "times"]

    # Calculate averages for each model and metric
    for (model, results) in model_results
        model_averages[model] = Dict{String, Float64}()
        
        for metric in metrics
            # Filter out NaN values before calculating the mean
            valid_values = filter(!isnan, collect(values(results[metric])))
            
            if !isempty(valid_values)
                model_averages[model][metric] = mean(valid_values)
            else
                model_averages[model][metric] = NaN
            end
        end
    end

    return model_averages
end

"""
    compute_result_averages(compiled_results::Dict{String, Dict{String, Dict{String, Float64}}})

Calculate the average values for cost, v_error, o_error and times for all models in the results dictionary

# Arguments
- `model_results::Dict{String, Dict{String, Dict{String, Float64}}}`: The dictionary of models to be analyzed.
- This `model_results` can be obtained by running the function `load_and_compile_results`.

# Returns
- `Dict{String, Dict{String, Float64}}`: A dictionary that maps the model to a metric to a value
"""
function compute_result_averages(compiled_results)
    model_averages = Dict{String, Dict{String, Float64}}()

    model_types = ["AC", "DC", "Linear", "Logarithmic", "Quadratic"]

    for model in model_types
        times = Float64[]
        costs = Float64[]
        verrs = Float64[]
        oerrs = Float64[]

        for (case_name, case_data) in compiled_results
            if haskey(case_data, model)
                push!(times, case_data[model]["time"])
                push!(costs, case_data[model]["cost"])
                push!(verrs, case_data[model]["v_error"])
                push!(oerrs, case_data[model]["o_error"])
            end
        end

        model_averages[model] = Dict(
            "average_time"   => mean(times),
            "average_cost"   => mean(costs) * 100,
            "average_v_error"=> mean(verrs) * 100,
            "average_o_error"=> mean(oerrs) * 100
        )
    end

    return model_averages
end

"""
    find_infeasible_constraints(model::Model)

check the feasibility of the constraints in a model, returns a dictionary of infeasible constraints

# Arguments
- `model::Model`: The model to be checked for constraint violations

# Returns
- `[]`: An array of the infeasible constraints.
"""
function find_infeasible_constraints(model::Model)
    if termination_status(model) != MOI.LOCALLY_INFEASIBLE
        println("The model must be optimized and locally infeasible")
		return []
    end

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

"""
    find_bound_violations(model::Model)

Check lower and upper bounds of all variables in a model, returns a dictionary of violations with additional mismatch data

# Arguments
- `model::Model`: The model to be checked for vounds violations

# Returns
- `Dict{VariableRef, Tuple{Float64, Float64, Float64, Float64}}`: A dictionary where each key 
is a variable ref and the value is a tuple of data.
"""
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

################################################################################
# Example of loading results from a directory and graphing them
# Do not uncomment the following lines, they are for demonstration purposes only
################################################################################

# using MPOPF
# using Ipopt
# using JuMP
# using MathOptInterface
# using PowerModels
# using Serialization

# results_dir = "results_cleaned/"

# models = load_and_compile_models(results_dir) # Just for show, not used here

# results = load_and_compile_results(results_dir)

# results["AC"]["times"]["case9"] # Get the time for the AC model for case9
# results["Logarithmic"]["times"]["case9"] # Get the time for the Logarithmic model for case9
# results["Linear"]["times"]["case14"] # Get the time for the Logarithmic model for case14

# feasibility_graph, v_error_graph, o_error_graph, time_graph = load_and_graph_results(results_dir, false)


################################################################################
# Example of calculating averages for the linearization paper
# Do not uncomment the following lines, they are for demonstration purposes only
################################################################################

# using MPOPF
# using Ipopt
# using JuMP
# using MathOptInterface
# using PowerModels
# using Serialization

# results_dir = "results_cleaned/"

# feasibility_graph, v_error_graph, o_error_graph, time_graph, compiled_results = load_and_graph_results_recalculate(results_dir, false)

# using Serialization

# compiled_results

# # For saving the results to a file
# serialize("compiled_results_used_for_averages.jls", compiled_results)

# # For loading the results from a file
# compiled_results = deserialize("compiled_results_used_for_averages.jls")

# display(compiled_results)

# compiled_results["case14"]["Quadratic"] # Example of getting the results for the Quadratic model for case14

# using Statistics

# # Example usage of getting averages:
# model_avgs = compute_result_averages(compiled_results)
# display(model_avgs)
