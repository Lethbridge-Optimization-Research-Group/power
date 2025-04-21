using CSV, DataFrames
using PlotlyJS
using MPOPF

function plot_case_sweep_results(
    case_name::String,
    sweep_name::String,
    results_dir::String="./probabilistic-results",
    plots_dir::String="./probabilistic-plots"
)
    # Construct the input and output paths
    input_dir = joinpath(results_dir, case_name, sweep_name)
    input_file = joinpath(input_dir, "results.csv")
    output_dir = joinpath(plots_dir, case_name, sweep_name)
    
    # Check if the results file exists
    if !isfile(input_file)
        println("Results file not found: $input_file")
        return false
    end
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Read the CSV file
    results = CSV.read(input_file, DataFrame)
    
    # Get parameter name from the results (should be the first column)
    param_name = names(results)[1]
    
    # Filter out rows with infeasible or error status for plotting
    feasible_results = filter(row -> row.status == "feasible", results)
    
    if nrow(feasible_results) == 0
        println("No feasible results found in $input_file")
        return false
    end
    
    # Get generator bus columns (columns that start with "pg_bus_")
    pg_cols = filter(col -> startswith(string(col), "pg_bus_"), names(results))
    
    # Extract bus numbers from column names
    bus_numbers = [parse(Int, replace(string(col), "pg_bus_" => "")) for col in pg_cols]
    
    # Plot all three types of graphs
    plot_generator_outputs(feasible_results, param_name, pg_cols, bus_numbers, output_dir)
    plot_objective_function(feasible_results, param_name, output_dir)
    plot_total_generation(feasible_results, param_name, pg_cols, bus_numbers, output_dir)
    
    println("Plots for $case_name $sweep_name saved to $output_dir")
    return true
end

function plot_generator_outputs(results::DataFrame, param_name::String, pg_cols, bus_numbers, output_dir::String)
    # Create graph for generator outputs with standardized filename
    graph_location = joinpath(output_dir, "generator_outputs.html")
    gen_graph = Graph(graph_location)
    
    # Get x values
    x_values = results[:, Symbol(param_name)]
    
    # Add a scatter plot for each generator
    for (i, col) in enumerate(pg_cols)
        bus = bus_numbers[i]
        y_values = results[:, col]
        add_scatter(gen_graph, x_values, y_values, "Generator at Bus $bus", i)
    end
    
    # Create and save the plot
    create_plot(
        gen_graph, 
        "Generator Outputs vs $param_name", 
        uppercase(param_name), 
        "Generator Output (p.u.)"
    )
    save_graph(gen_graph)
end

function plot_objective_function(results::DataFrame, param_name::String, output_dir::String)
    # Create graph for objective function with standardized filename
    graph_location = joinpath(output_dir, "objective_function.html")
    obj_graph = Graph(graph_location)
    
    # Get x values and y values
    x_values = results[:, Symbol(param_name)]
    y_values = results[:, :objective]
    
    # Add a scatter plot for the objective function
    add_scatter(obj_graph, x_values, y_values, "Objective Value", 1)
    
    # Create and save the plot
    create_plot(
        obj_graph, 
        "Objective Function vs $param_name", 
        uppercase(param_name), 
        "Objective Value"
    )
    save_graph(obj_graph)
end

function plot_total_generation(results::DataFrame, param_name::String, pg_cols, bus_numbers, output_dir::String)
    # Create graph for total generation with standardized filename
    graph_location = joinpath(output_dir, "total_generation.html")
    total_graph = Graph(graph_location)
    
    # Get x values
    x_values = results[:, Symbol(param_name)]
    
    # Calculate total generation for each row
    total_gen = zeros(nrow(results))
    for col in pg_cols
        total_gen .+= results[:, col]
    end
    
    # Add a scatter plot for total generation
    add_scatter(total_graph, x_values, total_gen, "Total Generation", 1)
    
    # Create and save the plot
    create_plot(
        total_graph, 
        "Total Generation vs $param_name", 
        uppercase(param_name), 
        "Total Generation (p.u.)"
    )
    save_graph(total_graph)
end

function plot_all_results(
    results_dir::String="./probabilistic-results",
    plots_dir::String="./probabilistic-plots"
)
    # Check if the results directory exists
    if !isdir(results_dir)
        println("Results directory not found: $results_dir")
        return
    end
    
    # Get all case directories in the results directory
    case_dirs = filter(x -> isdir(joinpath(results_dir, x)), readdir(results_dir))
    
    # Define the sweep names we expect to find
    sweep_names = ["epsilon", "confidence_level", "variation_value"]
    
    # Track which plots were successfully created
    successful_plots = 0
    total_possible_plots = 0
    
    # Process each case and sweep
    for case_name in case_dirs
        case_path = joinpath(results_dir, case_name)
        
        # Get all sweep directories for this case
        sweep_dirs = filter(x -> isdir(joinpath(case_path, x)), readdir(case_path))
        
        for sweep_name in sweep_dirs
            total_possible_plots += 1
            println("Processing plots for $case_name/$sweep_name...")
            
            # Plot results for this case and sweep
            if plot_case_sweep_results(case_name, sweep_name, results_dir, plots_dir)
                successful_plots += 1
            end
        end
    end
    
    println("Created $successful_plots/$total_possible_plots plots successfully.")
    println("All plots saved to $plots_dir")
end

# Example usage:
# Plot results for a specific case and sweep
# plot_case_sweep_results("case14", "epsilon")

# Plot all results
plot_all_results()