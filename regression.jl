#############################################################################
# Note: This file is curently being used for examples, nothing is permanent
#############################################################################

using JuMP, Ipopt, Serialization
using MPOPF
using PowerModels
using CSV
using Tables
using GLM
using DataFrames

# using AC factory
factory = ACMPOPFModelFactory("./Cases/case300.m", Ipopt.Optimizer)

data = PowerModels.parse_file(factory.file_path)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

#collecting data
ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
bus = 1
ref[:gen]
ref[:bus]
branch = ref[:branch][bus]

#loading data from model
models = load_and_compile_models("./results_small_cases")

#grabbing va values from optimized model
va = value.(models["case300"]["AC"].model[:va])

#creating arrays to store values
BranchArray = []
AngleDiffArray = []

#creating an index array and index value for branches
indexArray = []
num = 0

t=1

#adding values to each array for every branch
for (i, branch) in ref[:branch]
    va_fr = va[t, branch["f_bus"]]
    va_to = va[t, branch["t_bus"]]

    push!(BranchArray, i)

    diff = (va_fr - va_to)^2
    push!(AngleDiffArray, diff)

    global num = num + 1
    push!(indexArray, num)
end

# Writing to CSV file
#----------------------------------------------------------------------------------------------

# Create a dictionary to store the data
output_data = Dict(
    "Branch" => BranchArray, #which branch
    "AngleDiffSquare" => AngleDiffArray #(vi-vj)^2
)

#Write to CSV file
CSV.write("./Output/case300.csv", output_data)
#----------------------------------------------------------------------------------------------

#Linear Regrssion
#----------------------------------------------------------------------------------------------
# Performing Linear Regression
model_data = DataFrame(index = Float64.(indexArray), AngleDiff = Float64.(AngleDiffArray))
linear_model = lm(@formula(AngleDiff ~ index), model_data)

# Get the fitted values (predicted AngleDiff)
predicted_AngleDiff = predict(linear_model, model_data)
#----------------------------------------------------------------------------------------------

#Add values to a graph
graph = Graph("Output/case300AngleDiffSquare.html")
add_scatter(graph, indexArray, AngleDiffArray, "trace 1", 1, "markers")
add_scatter(graph, indexArray, predicted_AngleDiff, "Regression Line", 2, "lines")
create_plot(graph, "Difference Between Angle To and From Squared with Regression Line for Case 300", "Branch Index", "Angle Difference Squared")
save_graph(graph)
