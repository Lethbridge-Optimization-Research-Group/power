# a function for serializing data to a file
# updates the values if the file already exists
#
# output: Any - the data to be saved
# file_name: String - the name of the file to be saved
function save(file_name::String, output::Any)

	# Cretes the directory if it does not already exist
	mkpath(dirname(file_name))

	# check if the output directory exists
	if retreive(file_name) !== false
		data = retreive(file_name)
		data = merge(data, output)
		serialize(file_name, data)
	else
		serialize(file_name, output)
	end
end


# a function for deserializing data from a file, return false if not found
#
# file_name: String - the name of the file to be retrieved
function retreive(file_name::String)
	if isfile(file_name)
		data::Dict = deserialize(file_name)
		
		if data isa Dict
			return data
		else
			return false
		end
	else
		return false
	end
end


# Function for outputing string to file
#
# output: String - the output to be saved
# file_name: String - the name of the file to be saved
# show_date: Bool - whether to show the date in the file above the output
function output_to_file(data::String, file_name::String="", show_date::Bool=false)

	mkpath(dirname("output/" * file_name))

	# list all of the files in this directory
	files = readdir("output")

	# append to file
	if (file_name !== "")
		
		# check if it already exists
		if isfile("output/" * file_name)
			open("output/$(file_name)", "a") do io
				if show_date write(io, string(now())); write(io, "\n\n") end
				write(io, data)
				write(io, "\n\n")
			end
		else
			open("output/$(file_name)", "w") do io
				if show_date write(io, string(now())); write(io, "\n\n") end
				write(io, data)
				write(io, "\n\n")
			end
		end
	
	# create a new output file
	else

		# get the number of files in the directory
		num = length(files) + 1

		open("output/output_$(num).txt", "w") do io
			write(io, string(now()))
			write(io, "\n\n")
			write(io, data)
			write(io, "\n\n")
		end
	end
end

function get_random_scenarios(loads, min, max, num_scenarios, debug=false)
    scenarios = Dict()

    # Calculate bounds
    lower_bound = 1 - min
    upper_bound = 1 + max

    if debug
        println("Sampling diagnostics:")
        println("Bounds: [$lower_bound, $upper_bound]")
        println("Original loads: $loads")
        println()
    end

    for s in 1:num_scenarios
        scenario = Dict()
        for (i, load) in loads
            # Generate random multiplier
            multiplier = rand() * (upper_bound - lower_bound) + lower_bound
            scenario[i] = load * multiplier
            
            if debug
                println("Scenario $s, Load $i:")
                println("  Original Value: $load")
                println("  Multiplier: $multiplier")
                println("  New Value: $(scenario[i])")
                println()
            end
        end
        scenarios[s] = scenario
    end

    return scenarios

end



