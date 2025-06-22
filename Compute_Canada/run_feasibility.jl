using PowerModels
using Ipopt
using Serialization
using JuMP
using MPOPF

# Get the case file and model type from command-line arguments
case_file = ARGS[1]
model_type = ARGS[2]

function run_model(case_file, model_type)
    if model_type == "ac"
        factory = ACMPOPFModelFactory(case_file, Ipopt.Optimizer)
        compute_and_save_feasibility(factory, case_file)
    elseif model_type == "dc"
        factory = DCMPOPFModelFactory(case_file, Ipopt.Optimizer)
        compute_and_save_feasibility(factory, case_file)
    elseif model_type == "lin1"
        factory = LinMPOPFModelFactory(case_file, Ipopt.Optimizer)
        compute_and_save_feasibility(factory, case_file, Lin1)
    elseif model_type == "lin2"
        factory = LinMPOPFModelFactory(case_file, Ipopt.Optimizer)
        compute_and_save_feasibility(factory, case_file, Lin2)
    elseif model_type == "lin3"
        factory = LinMPOPFModelFactory(case_file, Ipopt.Optimizer)
        compute_and_save_feasibility(factory, case_file, Lin3)
    else
        error("Unknown model type: $model_type")
    end
end

# Run the specified model
run_model(case_file, model_type)
