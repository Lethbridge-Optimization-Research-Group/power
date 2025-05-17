using CSV, DataFrames, Random

function safe_parse_float(s::AbstractString)
    # Return the cell as a float, if none, input mising
    try
        return parse(Float64, s)
    catch
        return missing
    end
end

function parse_power_system_csv(file_path::String, matpower_file_path::String)
    # Get CSV content, compare CSV case name with matpower case name
    csv_content = read(file_path, String)
    lines = split(csv_content, '\n')
    csv_case_name = strip(lines[1])

    mat_power_case_name = basename(matpower_file_path)
    mat_power_case_name = replace(mat_power_case_name, ".m" => "")

    if csv_case_name != mat_power_case_name
        error("CSV case name ($csv_case_name) does not match the loaded MATPOWER case ($mat_power_case_name)")
    end

    # Read the entire CSV file into a DataFrame
    df = CSV.read(IOBuffer(join(lines[2:end], '\n')), DataFrame, header=1, skipto=2)
    # Initialize the output structures
    ramping_data = Dict{String,Any}()
    demands = Vector{Dict{Int,Float64}}()

    # Find the row where bus data starts
    bus_data_start = findfirst(x -> x == "#bus_data", df[!, 1])

    # Process generator data
    gen_data = df[1:bus_data_start-1, :]
    ramping_data["gen_id"] = [safe_parse_float(x) for x in gen_data[!, 1] if x != "gen_id"]
    ramping_data["ramp_limits"] = Dict{Int,Float64}()
    ramping_data["costs"] = Dict{Int,Float64}()
    
    # Convert ramp limits and costs to dictionaries indexed by gen_id
    for i in 1:length(ramping_data["gen_id"])
        gen_id = Int(ramping_data["gen_id"][i])
        ramping_data["ramp_limits"][gen_id] = safe_parse_float(gen_data[i+1, 2])
        ramping_data["costs"][gen_id] = safe_parse_float(gen_data[i+1, 3])
    end

    # Process bus data
    bus_data = df[bus_data_start+1:end, :]
    
    # Extract actual bus IDs from the first column
    bus_ids = [parse(Int, x) for x in bus_data[!, 1] if x != "bus_id" && !ismissing(x)]
    
    # Process demand data for each time period
    for t in 2:size(bus_data, 2)  # Start from column 2 (first time period)
        period_demands = Dict{Int,Float64}()
        for (idx, bus_id) in enumerate(bus_ids)
            # Get the demand value, default to 0.0 if missing
            demand_val = safe_parse_float(bus_data[idx+1, t])
            period_demands[bus_id] = ismissing(demand_val) ? 0.0 : demand_val
        end
        push!(demands, period_demands)
    end

    return ramping_data, demands
end


function generate_power_system_csv(data::Dict, output_dir::String, num_periods::Int=24)
    # Extract case name
    case_name = basename(data["name"])
    case_name = replace(case_name, ".m" => "")

    # Create a filename
    output_file = joinpath(output_dir, "$(case_name)_rampingData.csv")

    # Calculate total generation capacity
    total_generation_capacity = 0.0
    gen_data = []
    for (_, gen) in data["gen"]
        pmax = get(gen, "pmax", 0.0)
        total_generation_capacity += pmax

        # Calculate ramping limit as percentage of generator output
        ramp_percent = rand(90:100)  # Random percentage between 5% and 50%
        ramp_limit = pmax * (ramp_percent / 100)

        # Generate random ramping cost
        ramp_cost = rand(10:70) ### RAMP COST ### 

        push!(gen_data, (
            gen["index"],
            round(ramp_limit, digits=2),
            round(ramp_cost, digits=2)
        ))
    end
    sort!(gen_data, by=x -> x[1])

    # Extract bus demands and create a mapping of actual bus IDs
    demand_dict = Dict{Int,Float64}()
    bus_ids = Int[]  # Store actual bus IDs in order

    # First, collect all bus IDs from the bus data
    for (_, bus) in data["bus"]
        push!(bus_ids, bus["bus_i"])
    end
    sort!(bus_ids)  # Ensure buses are in order

    # Then collect the demands
    total_initial_demand = 0.0
    for (_, load) in data["load"]
        bus_id = load["load_bus"]
        pd = get(load, "pd", 0.0)
        demand_dict[bus_id] = get(demand_dict, bus_id, 0.0) + pd
        total_initial_demand += pd
    end

    # Safety margin (95% of total capacity)
    max_allowable_demand = total_generation_capacity * 0.95

    # If initial demand exceeds capacity, scale it down
    if total_initial_demand > max_allowable_demand
        scaling_factor = max_allowable_demand / total_initial_demand
        for bus_id in keys(demand_dict)
            demand_dict[bus_id] *= scaling_factor
        end
    end

    # Create initial demands only for actual buses
    initial_demand = [get(demand_dict, bus_id, 0.0) for bus_id in bus_ids]

    # Generate random variations for additional time periods
    Random.seed!()  # Input a seed if you like for reproducibility
    demands = [initial_demand]

    for _ in 2:num_periods
        variation = rand(length(bus_ids)) * 0.2 .- 0.1  ### RAMP VARIATION ### 
        new_demand = initial_demand .* (1 .+ variation)
        new_demand = max.(new_demand, 0)  # Ensure non-negative demands

        # Check if total demand exceeds capacity and scale if necessary
        total_new_demand = sum(new_demand)
        if total_new_demand > max_allowable_demand
            scaling_factor = max_allowable_demand / total_new_demand
            new_demand *= scaling_factor
        end

        push!(demands, new_demand)
    end

    # Create the CSV content
    csv_content = IOBuffer()
    println(csv_content, case_name)
    println(csv_content, "#gen_data")
    println(csv_content, "gen_id,ramp_limits,costs")
    for (index, ramp, cost) in gen_data
        println(csv_content, "$index,$ramp,$cost")
    end
    println(csv_content, "#bus_data")
    print(csv_content, "bus_id")
    for i in 1:num_periods
        print(csv_content, ",T$i")
    end
    println(csv_content)

    # Only output data for buses that exist in the system
    for (idx, bus_id) in enumerate(bus_ids)
        print(csv_content, bus_id)
        for period in 1:num_periods
            print(csv_content, ",", demands[period][idx])
        end
        println(csv_content)
    end

    # Write to file
    open(output_file, "w") do f
        write(f, String(take!(csv_content)))
    end

    println("CSV file generated successfully: $output_file")
    println("Total generation capacity: ", round(total_generation_capacity, digits=2))
    println("Maximum allowable demand: ", round(max_allowable_demand, digits=2))
    return output_file
end