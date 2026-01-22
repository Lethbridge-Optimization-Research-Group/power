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
# enter date as yyyy-mm-dd
using CSV, DataFrames

using CSV, DataFrames

function get_date_percentages(file_path, date)
    # Read data
    df = CSV.read(file_path, DataFrame; skipto=5, header=false)
    rename!(df, [:Date, :Hour, :Market_Demand, :Ontario_Demand])

    # Clean date column
    df.Date = strip.(string.(df.Date))

    # Filter for specific date
    day_data = df[df.Date .== date, :]

    if nrow(day_data) == 0
        error("No data found for date $date")
    end

    # Get Ontario Demand for each hour (0 if missing)
    hourly_demand = [
        isempty(day_data[day_data.Hour .== i, :Ontario_Demand]) ?
            0 :
            day_data[day_data.Hour .== i, :Ontario_Demand][1]
        for i in 1:24
    ]

    # Compute percentages of that day's peak
    max_demand = maximum(hourly_demand)
    return hourly_demand ./ max_demand
end
