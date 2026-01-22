using CSV, DataFrames, Random

function safe_parse_float(s::AbstractString)
    # Return the cell as a float, if none, input mising
    try
        return parse(Float64, s)
    catch
        return missing
    end
end

# Do not leave cells blank, insert 0 for busses that don't demand power
function parse_power_system_csv(file_path::String, matpower_file_path::String)
    # Get CSV content, compare CSV case name with matpoewr case name
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
    ramping_data = Dict{String, Vector{Float64}}()
    demands = Vector{Vector{Float64}}()

    # Find the row where bus data starts
    bus_data_start = findfirst(x -> x == "#bus_data", df[!, 1])

    # Process generator data
    gen_data = df[1:bus_data_start-1, :]
    ramping_data["gen_id"] = [safe_parse_float(x) for x in gen_data[!, 1] if x != "gen_id"]
    ramping_data["ramp_limits"] = [safe_parse_float(x) for x in gen_data[!, 2] if x != "ramp_limits"]
    ramping_data["costs"] = [safe_parse_float(x) for x in gen_data[!, 3] if x != "costs"]

    # Process bus data
    bus_data = df[bus_data_start+1:end, 2:end]
    for col in names(bus_data)
        push!(demands, filter(!ismissing, [safe_parse_float(x) for x in bus_data[!, col]]))
    end

    return ramping_data, demands
end

function generate_power_system_csv(data::Dict, output_dir::String, num_periods::Int=24)
    # Extract case name
    case_name = basename(data["name"])
    case_name = replace(case_name, ".m" => "")

    # Create a filename
    output_file = joinpath(output_dir, "$(case_name)_rampingData.csv")

    # Extract generator data
    gen_data = []
    for (_, gen) in data["gen"]
        # Calculate ramping limit as percentage of generator output
        pmax = get(gen, "pmax", 0.0)
        ramp_percent = rand(50:100)  # Random percentage between 5% and 50%
        ramp_limit = pmax * (ramp_percent / 100)
        
        # Generate random ramping cost
        ramp_cost = rand(100:700)  # Random cost between 5 and 20

        push!(gen_data, (
            gen["index"],
            round(ramp_limit, digits=2),
            round(ramp_cost, digits=2)  # This is now the ramp cost
        ))
    end
    sort!(gen_data, by = x -> x[1])

    # Extract initial bus demands
    demand_dict = Dict{Int, Float64}()
    for (_, load) in data["load"]
        bus_id = load["load_bus"]
        pd = get(load, "pd", 0.0)
        demand_dict[bus_id] = get(demand_dict, bus_id, 0.0) + pd
    end

    # Create a vector of demands, ensuring we have a value for each bus
    num_buses = length(data["bus"])
    initial_demand = [get(demand_dict, i, 0.0) for i in 1:num_buses]

    # Generate random variations for additional time periods
    Random.seed!()  # Input a seed if you like for reproducibility
    demands = [initial_demand]
    for _ in 2:num_periods
        variation = rand(num_buses) * 0.2 .- 0.1  # Random variation between -20% and +20%
        new_demand = initial_demand .* (1 .+ variation)
        push!(demands, new_demand)  # Maybe ensure non-negative demands
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
    for bus in 1:num_buses
        print(csv_content, bus)
        for period in 1:num_periods
            print(csv_content, ",", round(demands[period][bus], digits=3))
        end
        println(csv_content)
    end

    # Write to file
    open(output_file, "w") do f
        write(f, String(take!(csv_content)))
    end

    println("CSV file generated successfully: $output_file")
    return output_file
end