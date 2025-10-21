using PowerModels, JuMP, Ipopt, Gurobi, PlotlyJS
using MPOPF

file_path = "Cases/case14.m"

ac_factory = ACMPOPFModelFactory(file_path, Ipopt.Optimizer)

my_ac_model = create_model(ac_factory; time_periods=3, factors = [1.0, 0.98, 1.03], ramping_cost=7)

optimize_model(my_ac_model)