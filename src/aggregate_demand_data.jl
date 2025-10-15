using CSV, DataFrames

function parse_csv_data(file_path::String)
    # Read data, skip the metadata lines, and ignore headers entirely
    df = CSV.read(file_path, DataFrame; skipto=5, header=false)

    # Rename columns to something predictable
    rename!(df, [:Date, :Hour, :Market_Demand, :Ontario_Demand])

    # Convert numeric columns just in case they were read as strings
    #df.Hour = parse.(Int, df.Hour)
    #df.Market_Demand = parse.(Float64, df.Market_Demand)
    #df.Ontario_Demand = parse.(Float64, df.Ontario_Demand)

    # Compute average demand per hour
    hourly_avg = combine(groupby(df, :Hour),
        :Market_Demand => mean => :Avg_Market_Demand,
        :Ontario_Demand => mean => :Avg_Ontario_Demand)

    return hourly_avg
end

function get_hourly_average(data)
    hourly_averages = []
    for i in 1:24
        push!(hourly_averages, data[data.Hour .== i, :Avg_Ontario_Demand][1])
    end
    return hourly_averages
end

function percentages_of_max_demand(hourly_averages)
    highest_demand = maximum(hourly_averages)
    percentages = []
    for hour in hourly_averages
        push!(percentages, hour/highest_demand)
    end
    return percentages
end