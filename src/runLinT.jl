using MPOPF
using Ipopt
using JuMP
using PowerModels
const PM = PowerModels

file_path = "/home/prottasha-dcruze/UniversityProjects/OpticalPowerFlow/power/Cases/case14.m"

data = PowerModels.parse_file(file_path)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

ac_factory = LinTMPOPFModelFactory(file_path, Ipopt.Optimizer)

My_AC_model = create_model_demand(ac_factory; i = 0)

optimize_model(My_AC_model)
