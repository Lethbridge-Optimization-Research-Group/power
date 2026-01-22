# using PowerModels
using JuMP
using Ipopt
include("MPOPF.jl")
using .MPOPF

# We define the file path of the case we want to solve
file_path = "./Cases/case14.m"

# To create AC model we need to first define a AC factory
# It is done with the following function
# Takes in two parameters, the fille path for the case we want to solve
# and the optimizer we want to use, Ipopt or Gurobi
ac_factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)
# After creating our factory we pass it to our create model function
my_ac_model = create_model(ac_factory)
# Once we have our model we just optimize
# This will print the Minimum Cost
MPOPF.optimize_model(my_ac_model)



