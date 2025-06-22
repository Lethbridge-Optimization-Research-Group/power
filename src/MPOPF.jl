"""
# MPOPF
A module for Multi-Period Optimal Power Flow (MPOPF) modeling and optimization.
This module provides tools to create, optimize, and analyze MPOPF models using various formulations including AC, DC, and linearized versions.
## Main Features
- Create MPOPF models using different factory types (AC, DC, Linear, etc.)
- Optimize MPOPF models
- Handle uncertainty in MPOPF models
- Visualize optimization results
"""
module MPOPF
    using PowerModels, JuMP, Dates, Serialization, PlotlyJS, Ipopt
    using Distributions, Statistics
    using LinearAlgebra, Random

    # Exporting these functions from the module so we dont have to prefix them with MPOPF.

    # Export of this file
    export create_model, create_search_model, optimize_model, ACMPOPFModelFactory, DCMPOPFModelFactory, optimize_model_with_plot, LinMPOPFModelFactory, NewACMPOPFModelFactory, DCMPOPFSearchFactory, create_model_check_feasibility, get_ref

    # Export of implementation_uncertainty.jl
    export generate_random_load_scenarios, setup_demand_distributions, sample_demand_scenarios, return_loads
    export generate_correlated_scenarios, verify_scenario_correlation, analyze_scenario_statistics, calculate_load_correlations, test_concrete_solution

    # Export of Graphing_class.jl functions
    # TODO: Need to have proper documentation for the Graphing class
    export Graph, add_scatter, create_plot, save_graph, display_graph

    # Export of misc.jl functions
    # TODO: Need to document these functions
    # TODO: Need to rename these functions as well, for example, save is confusing
    # TODO: After renaming them need to also rename the places they are used
    export generate_load_scenarios, save, retreive, output_to_file, get_random_scenarios

    # Export of graphing_feasibility.jl
    export perform_feasibility

    # Export of compute_and_save_feasibility.jl
    export compute_and_save_feasibility, load_and_graph_results, load_and_compile_results, compute_result_averages, find_infeasible_constraints, find_bound_violations, load_and_compile_models

    # Export of search_functions.jl
    export create_initial_feasible_solution, check_solution, adjust_to_meet_demand, apply_ramping_constraints, optimize_solution, calculate_total_cost,
    is_feasible_solution, decomposed_mpopf_local_search, decomposed_mpopf_demand_search, check_ramping_limits, check_demands_met, check_demands_met_output,
    sort_time_periods_by_demand, adjust_adjacent_periods, create_decomposed_mpopf_model, create_initial_random_solution, generate_bounded_random_solution, 
    adjust_solution_for_constraints, verify_solution_feasibility

    # Export of rampingCSVimplementation.jl
    export safe_parse_float, parse_power_system_csv, generate_power_system_csv


    # create enum for linear models
    @enum MODEL_TYPE begin
        Undef=0
        Lin1=1
        Lin2=2
        Lin3=3
    end

    export MODEL_TYPE, Lin1, Lin2, Lin3

##############################################################################################
# Factory Structs
# They are used as parameters so that Julias multiple dispatch knows which functions to call
##############################################################################################

    # Abstract type as a base so that we can use this type as a parameter in functions
    """
        AbstractMPOPFModelFactory
    An abstract type serving as a base for all MPOPF model factories.
    """
    abstract type AbstractMPOPFModelFactory end

    # This struct "inherits" from PowerFlowModelFactory
    """
        ACMPOPFModelFactory <: AbstractMPOPFModelFactory
    Factory for creating AC MPOPF models.
    # Fields
    - `file_path::String`: Path to the input data file.
    - `optimizer::Type`: The optimizer to be used (e.g., Ipopt.Optimizer).
    """
    mutable struct ACMPOPFModelFactory <: AbstractMPOPFModelFactory
        file_path::String
        optimizer

        function ACMPOPFModelFactory(file_path::String, optimizer)
            return new(file_path, optimizer)
        end
    end

    # This struct "inherits" from PowerFlowModelFactory
    """
        DCMPOPFModelFactory <: AbstractMPOPFModelFactory
    Factory for creating DC MPOPF models.
    # Fields
    - `file_path::String`: Path to the input data file.
    - `optimizer::Type`: The optimizer to be used.
    """
    mutable struct DCMPOPFModelFactory <: AbstractMPOPFModelFactory
        file_path::String
        optimizer::Type

        function DCMPOPFModelFactory(file_path::String, optimizer::Type)
            return new(file_path, optimizer)
        end
    end

    """
        LinMPOPFModelFactory <: AbstractMPOPFModelFactory
    Factory for creating linearized MPOPF models.
    If this factory is used then `model_type` needs to be specified
    in the `create_model()` function.
    # Fields
    - `file_path::String`: Path to the input data file.
    - `optimizer::Type`: The optimizer to be used.
    """
    mutable struct LinMPOPFModelFactory <: AbstractMPOPFModelFactory
        file_path::String
        optimizer::Type

        function LinMPOPFModelFactory(file_path::String, optimizer::Type)
            return new(file_path, optimizer)
        end
    end

    """
        NewACMPOPFModelFactory <: AbstractMPOPFModelFactory
    Factory for creating new AC MPOPF models.
    This is only used when computing for feasibility
    # Fields
    - `file_path::String`: Path to the input data file.
    - `optimizer::Type`: The optimizer to be used.
    """
    mutable struct NewACMPOPFModelFactory <: AbstractMPOPFModelFactory
        file_path::String
        optimizer::Type

        function NewACMPOPFModelFactory(file_path::String, optimizer::Type)
            return new(file_path, optimizer)
        end
    end
    ```
        TODO: Document
    ```
    mutable struct DCMPOPFSearchFactory <: AbstractMPOPFModelFactory
        file_path::String
        optimizer::Type

        function DCMPOPFSearchFactory(file_path::String, optimizer::Type)
            return new(file_path, optimizer)
        end
    end

##############################################################################################
# Concrete Model Structs
# They are used as objects, passed around with variables that are specific to each model
##############################################################################################

    # Abstract type as a base so that we can use this type as a parameter in functions
    """
        AbstractMPOPFModel
    An abstract type serving as a base for all MPOPF model types.
    """
    abstract type AbstractMPOPFModel end

    # The actual PowerFlowModel struct that "inherits" from AbstractPowerFlowModel
    """
        MPOPFModel <: AbstractMPOPFModel
    Represents a Multi-Period Optimal Power Flow model.
    # Fields
    - `model::JuMP.Model`: The underlying JuMP model.
    - `data::Dict`: Dictionary containing the power system data.
    - `time_periods::Int64`: Number of time periods in the model.
    - `factors::Vector{Float64}`: Scaling factors for each time period.
    - `ramping_cost::Int64`: Cost associated with generator ramping.
    """
    mutable struct MPOPFModel <: AbstractMPOPFModel
        model::JuMP.Model
        data::Dict
        time_periods::Int64
        factors::Vector{Float64}
        ramping_cost::Int64

        function MPOPFModel(model::JuMP.Model, data::Dict, time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0)
            return new(model, data, time_periods, factors, ramping_cost)
        end
    end

    # Similar PowerFlowModel object but with an additional scenarios variable for uncertainty
    """
        MPOPFModelUncertainty <: AbstractMPOPFModel
    Represents a Multi-Period Optimal Power Flow model with uncertainty considerations.
    # Fields
    - `model::JuMP.Model`: The underlying JuMP model.
    - `data::Dict`: Dictionary containing the power system data.
    - `scenarios::Dict`: Dictionary of scenarios for uncertainty analysis.
    - `time_periods::Int64`: Number of time periods in the model.
    - `factors::Vector{Float64}`: Scaling factors for each time period.
    - `ramping_cost::Int64`: Cost associated with generator ramping.
    """
    mutable struct MPOPFModelUncertainty <: AbstractMPOPFModel
        model::JuMP.Model
        data::Dict
        scenarios::Dict
        time_periods::Int64
        factors::Vector{Float64}
        ramping_cost::Int64

        function MPOPFModelUncertainty(model::JuMP.Model, data::Dict, scenarios::Dict, time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0)
            return new(model, data, scenarios, time_periods, factors, ramping_cost)
        end
    end
    ```
        TODO: Document
    ```
    mutable struct MPOPFSearchModel <: AbstractMPOPFModel
        model::JuMP.Model
        data::Dict
        time_periods::Int64
        ramping_data::Dict
        demands::Vector{Vector{Float64}}

        function MPOPFSearchModel(model::JuMP.Model, data::Dict, time_periods::Int64, ramping_data::Dict, demands::Vector{Vector{Float64}})
            return new(model, data, time_periods, ramping_data, demands)
        end
    end

##############################################################################################
# Create Model Functions
# These functions return PowerFlowModel objects with or without uncertainty
##############################################################################################

    # Here we include our implementation files
    # They hold the implementations of the functions utilized in the create_model functions
    include("graphing_class.jl")
    include("implementation-ac.jl")
    include("implementation-dc.jl")
    include("implementation_uncertainty.jl")
    include("implementation-linear.jl")
    include("implementation-new_ac.jl")
    include("implementation-search_dc.jl")
    include("misc.jl")
    include("graphing_feasibility.jl")
    include("compute_and_save_feasibility.jl")
    include("search_functions.jl")
    include("search2.jl")
    include("rampingCSVimplementation.jl")

    # The first create_model fucntion creates a PowerFlowModel object
    # It creates the right model depending on the factory passed as the first paramenter
    # For Example: If the factory passed is an AC factory the function will return an AC model
    """
        create_model(factory::AbstractMPOPFModelFactory; time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0, model_type=undef)::MPOPFModel
    Create a Multi-Period Optimal Power Flow (MPOPF) model.
    The factory passed in defines the type of model that will be returned
    If `LinMPOPFModelFactory` is passed in as the factory then `model_type` needs to be specified.
    # Arguments
    - `factory`: The factory used to create the specific type of MPOPF model.
    - `time_periods`: Number of time periods to consider in the model. Default is 1.
    - `factors`: Scaling factors for each time period. Default is [1.0].
    - `ramping_cost`: Cost associated with ramping generation up or down. Default is 0.
    - `model_type`: Optional parameter to specify a particular model type. Default is undef.
    # Returns
    An `MPOPFModel` object representing the created MPOPF model.
    """
    function create_model(factory::AbstractMPOPFModelFactory; time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0, model_type=undef)::MPOPFModel
        data = PowerModels.parse_file(factory.file_path)
        PowerModels.standardize_cost_terms!(data, order=2)
        PowerModels.calc_thermal_limits!(data)

        model = JuMP.Model(factory.optimizer)

        power_flow_model = MPOPFModel(model, data, time_periods, factors, ramping_cost)

        set_model_variables!(power_flow_model, factory)
        set_model_objective_function!(power_flow_model, factory)
        model_type !== undef ? set_model_constraints!(power_flow_model, factory, model_type) : set_model_constraints!(power_flow_model, factory)

        return power_flow_model
    end

    # The second create_model fucntion creates a PowerFlowModelUncertainty object
    # Similarly it creates the right model depending on the factory passed as the first paramenter
    # However now we are passing the scenarios as parameters too
    """
        create_model(factory::AbstractMPOPFModelFactory, scenarios::Dict, time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0)::MPOPFModelUncertainty
    Create a Multi-Period Optimal Power Flow (MPOPF) model with uncertainty considerations.
    # Arguments
    - `factory`: The factory used to create the specific type of MPOPF model.
    - `scenarios`: Dictionary of scenarios for uncertainty analysis.
    - `time_periods`: Number of time periods to consider in the model. Default is 1.
    - `factors`: Scaling factors for each time period. Default is [1.0].
    - `ramping_cost`: Cost associated with ramping generation up or down. Default is 0.
    # Returns
    An `MPOPFModelUncertainty` object representing the created MPOPF model with uncertainty.
    """
    function create_model(factory::AbstractMPOPFModelFactory, scenarios::Dict, mismatch_costs::Tuple{Float64,Float64}=(10000.0, 10000.0), time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0)::MPOPFModelUncertainty
        data = PowerModels.parse_file(factory.file_path)
        PowerModels.standardize_cost_terms!(data, order=2)
        PowerModels.calc_thermal_limits!(data)

        mu_plus_cost = mismatch_costs[1]
        mu_minus_cost = mismatch_costs[2]

        model = JuMP.Model(factory.optimizer)

        power_flow_model = MPOPFModelUncertainty(model, data, scenarios, time_periods, factors, ramping_cost)

        set_model_variables!(power_flow_model, factory)
        set_model_uncertainty_variables!(power_flow_model)
        set_model_uncertainty_objective_function!(power_flow_model, factory, mu_plus_cost, mu_minus_cost)
        set_model_uncertainty_constraints!(power_flow_model, factory)

        return power_flow_model
    end

    # A new create model to create a secondary model that asseses how feasible the first solution was
    # It uses the pg values from a previous model and fixes it, then asses if it works referencing AC OPF
    """
        create_model_check_feasibility(factory::NewACMPOPFModelFactory, new_pg=false, new_qg=false, v=false, theta=false, time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0)::MPOPFModel
    Create a secondary model to assess the feasibility of a solution
    when comparing to the AC model.
    # Arguments
    - `factory`: The factory used to create the specific type of MPOPF model.
    - `new_pg`: Fixed values for active power generation, or `false` to skip fixing.
    - `new_qg`: Fixed values for reactive power generation, or `false` to skip fixing.
    - `v`: Fixed values for bus voltages, or `false` to skip fixing.
    - `theta`: Fixed values for bus angles, or `false` to skip fixing.
    - `time_periods`: Number of time periods to consider in the model. Default is 1.
    - `factors`: Scaling factors for each time period. Default is [1.0].
    - `ramping_cost`: Cost associated with ramping generation up or down. Default is 0.
    # Returns
    An `MPOPFModel` object representing the created MPOPF model.
    """
    function create_model_check_feasibility(factory::NewACMPOPFModelFactory, new_pg=false, new_qg=false, v=false, theta=false, time_periods::Int64=1, factors::Vector{Float64}=[1.0], ramping_cost::Int64=0)::MPOPFModel
        data = PowerModels.parse_file(factory.file_path)
        PowerModels.standardize_cost_terms!(data, order=2)
        PowerModels.calc_thermal_limits!(data)

        model = JuMP.Model(factory.optimizer)

        power_flow_model = MPOPFModel(model, data, time_periods, factors, ramping_cost)

        set_model_variables!(power_flow_model, factory)

        # fix values that have been declared
        if (new_pg !== false) fix.(power_flow_model.model[:pg], new_pg; force=true) end
        if (new_qg !== false) fix.(power_flow_model.model[:qg], new_qg; force=true) end
		if (v !== false) fix.(power_flow_model.model[:v], v; force=true) end
		if (theta !== false) fix.(power_flow_model.model[:theta], theta; force=true) end

        set_model_objective_function!(power_flow_model, factory)
        set_model_constraints!(power_flow_model, factory)

        return power_flow_model
    end


    ```
        TODO: Document
    ```
    function create_search_model(factory::AbstractMPOPFModelFactory, time_periods::Int64, ramping_data::Dict, demands::Vector{Vector{Float64}})::MPOPFSearchModel
        data = PowerModels.parse_file(factory.file_path)
        PowerModels.standardize_cost_terms!(data, order=2)
        PowerModels.calc_thermal_limits!(data)

        model = JuMP.Model(factory.optimizer)

        power_flow_model = MPOPFSearchModel(model, data, time_periods, ramping_data, demands)

        set_model_variables!(power_flow_model, factory)
        set_model_objective_function!(power_flow_model, factory)
        set_model_constraints!(power_flow_model, factory)

        return power_flow_model
    end

##############################################################################################
# Optimization functions
##############################################################################################

    # This function simply optimizes any model given as a parameter
    # It prints the Optimial cost
    """
        optimize_model(model::AbstractMPOPFModel)
    Optimize the given MPOPF model and print the optimal cost.
    
    !!! note
        If the model being optimized is of type `MPOPFModelUncertainty`
        then extra computation will be executed to determine if there are
        any unbalaced power at each bus for every scenario
        when consitering mu_plus and mu_minus.
        That is if `sum_of_mu_plus` and `mu_minus` are both > 0.01
        at any given bus then an error message is printed.
    # Arguments
    - `model`: The MPOPF model to optimize.
    """
    function optimize_model(model::AbstractMPOPFModel)
        optimize!(model.model)
        optimal_cost = objective_value(model.model)
        println("Optimal Cost: ", optimal_cost)
        println()

        if isa(model, MPOPFModelUncertainty)
            data = model.data
            T = model.time_periods
            S = length(model.scenarios)
            ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
            error_check = 0

            for t in 1:T    
                for s in 1:S
                    for b in keys(ref[:bus])
                        sum_of_mu_plus = 0
                        mu_minus = 0
                        sum_of_mu_plus = sum(value(model.model[:mu_plus][t, g, s]) for g in ref[:bus_gens][b]; init = 0)
                        mu_minus = value(model.model[:mu_minus][t, b, s])
                        if sum_of_mu_plus >= 0.01 && mu_minus >= 0.01
                            error_check = error_check + 1
                            println("###############")
                            println("#### Error ####")
                            println("###############")

                            println("Scenario: $s Bus: $b")
                            println("mu_plus: $sum_of_mu_plus")
                            println("mu_minus: $mu_minus")
                            println()
                        end
                    end
                end
            end
            if error_check == 0
                println("No mu_plus and mu_minus errors found")
                println()
            else
                println("Found $error_check error(s)")
                println()
            end
        end
    end

    # Optimized and graphs the given model using a callback function
    """
        optimize_model_with_plot(model::AbstractMPOPFModel)
    Optimize the given MPOPF model, print the optimal cost, 
    and generate a PlotlyJS graph/plot of the optimization process.
    
    !!! note
        if `time_periods = 1` then the graph will represent the iterations taken by the solver
        otherwize the graph will represent each time period.
    # Arguments
    - `model`: The MPOPF model to optimize and plot.
    """
    function optimize_model_with_plot(model::AbstractMPOPFModel)
        T = model.time_periods

        if T == 1
            error("Graphing for Time Periods = 1 not supported")
            # objective_values = Float64[]
            # iterations = Int[]

            # function ipopt_callback(
            #     alg_mod::Cint, iter_count::Cint, obj_value::Float64,
            #     inf_pr::Float64, inf_du::Float64, mu::Float64,
            #     d_norm::Float64, regularization_size::Float64,
            #     alpha_du::Float64, alpha_pr::Float64, ls_trials::Cint
            # )
            #     push!(objective_values, obj_value)
            #     push!(iterations, iter_count)
            #     return true  # Return true to continue the optimization
            # end

            # # Note: The callback does not work without a new package
            # # https://github.com/jump-dev/Gurobi.jl#callbacks
            # function gurobi_callback(cb_data, where)
            #     if where == GRB_CB_MIP
            #         iteration = Ref{Cint}()
            #         GRBcbget(cb_data, where, GRB_CB_MIP_NODCNT, iteration)

            #         objbst = Ref{Cdouble}()
            #         GRBcbget(cb_data, where, GRB_CB_MIP_OBJBST, objbst)

            #         push!(iterations, iteration[])
            #         push!(objective_values, objbst[])

            #         println("Iteration: $(iteration[]), Best Obj: $(objbst[])")
            #     end
            # end

            # if solver_name(model.model) == "Ipopt"
            #     MOI.set(model.model, Ipopt.CallbackFunction(), Ipopt.Optimizer)
            # elseif solver_name(model.model) == "Gurobi"
            #     error("At the moment there is no graphing for gurobi if Time_periods = 1")
            #     # MOI.set(model.model, MOI.RawOptimizerAttribute("LazyConstraints"), 1)
            #     # MOI.set(model.model, Gurobi.CallbackFunction(), gurobi_callback)
            #     # if MOI.get(model.model, MOI.NumberOfVariables()) > 0
            #     #     println("Callback has been set for Gurobi")
            #     # else
            #     #     println("No variables in the model. Callback may not be triggered.")
            #     # end
            # else
            #     error("Optimizer must be either Ipopt or Gurobi")
            # end

            # optimize!(model.model)
            # optimal_cost = objective_value(model.model)
            # println("Optimal Cost: ", optimal_cost)

            # # Plotting Code
            # trace = scatter(
            #     x=iterations, y=objective_values,
            #     mode="lines+markers",
            #     name="Objective Cost",
            #     marker_color="blue",
            #     hoverinfo="x+y", # Ensure hover displays both x and y values
            #     hovertemplate="%{x}, %{y:.2f}<extra></extra>" # Custom hover text format
            # )

            # layout = Layout(
            #     title="Plotting Objective Cost against Solver Iterations",
            #     xaxis=attr(title="Iterations", tickangle=-45, tickmode="linear", tick0=0, dtick=1),
            #     yaxis=attr(title="Objective Cost", hoverformat=".2f"),
            #     showlegend=true
            # )

            # Plot = plot([trace], layout)

            # return Plot
        else
            optimize!(model.model)
            optimal_cost = objective_value(model.model)
            println("Optimal Cost: ", optimal_cost)

            # Extracting cost components
            ref = PowerModels.build_ref(model.data)[:it][:pm][:nw][0]
            pg = model.model[:pg]
            ramp_up = model.model[:ramp_up]
            ramp_down = model.model[:ramp_down]
            ramping_cost = model.ramping_cost

            ramping_cost_per_period = [sum(ramping_cost * (value(ramp_up[t, g]) + value(ramp_down[t, g])) for g in keys(ref[:gen])) for t in 2:T]
            ramping_up_per_period = [sum(ramping_cost * (value(ramp_up[t, g])) for g in keys(ref[:gen])) for t in 2:T]
            ramping_down_per_period = [sum(ramping_cost * (value(ramp_down[t, g])) for g in keys(ref[:gen])) for t in 2:T]

            # Adjusting ramping cost array to match periods (adding zero for first period)
            ramping_cost_per_period = [0.0; ramping_cost_per_period]
            ramping_up_per_period = [0.0; ramping_up_per_period]
            ramping_down_per_period = [0.0; ramping_down_per_period]

            cost_per_period_with_ramping_to_that_period = [sum(ref[:gen][g]["cost"][1]*value(pg[t,g])^2 + ref[:gen][g]["cost"][2]*value(pg[t,g]) + ref[:gen][g]["cost"][3] for g in keys(ref[:gen])) + ramping_cost_per_period[t] for t in 1:T]
            cost_per_period_no_ramping = [sum(ref[:gen][g]["cost"][1]*value(pg[t,g])^2 + ref[:gen][g]["cost"][2]*value(pg[t,g]) + ref[:gen][g]["cost"][3] for g in keys(ref[:gen])) for t in 1:T]

            rolling_cost_per_period = zeros(Float64, T)  # Initialize the array with zeros
            rolling_cost_per_period[1] = cost_per_period_with_ramping_to_that_period[1]
            for t in 2:T
                rolling_cost_per_period[t] = rolling_cost_per_period[t-1] + cost_per_period_with_ramping_to_that_period[t]
            end

            # Plotting costs
            trace_cost_per_period_no_ramping = scatter(
                x=1:T,
                y=cost_per_period_no_ramping,
                mode="lines+markers",
                name="Objective Cost no Ramping",
                marker_color="black"
            )

            trace_cost_per_period_with_ramping_to_that_period = scatter(
                x=1:T,
                y=cost_per_period_with_ramping_to_that_period,
                mode="lines+markers",
                name="Objective Cost With Ramping",
                marker_color="blue"
            )

            trace_ramping_cost_per_period = scatter(
                x=1:T,
                y=ramping_cost_per_period,
                mode="lines+markers",
                name="Ramping Cost",
                marker_color="red"
            )

            trace_ramping_up_per_period = scatter(
                x=1:T,
                y=ramping_up_per_period,
                mode="lines+markers",
                name="Ramping Up",
                marker_color="green"
            )

            trace_ramping_down_per_period = scatter(
                x=1:T,
                y=ramping_down_per_period,
                mode="lines+markers",
                name="Ramping Down",
                marker_color="orange"
            )

            trace_rolling_cost_per_period = scatter(
                x=1:T,
                y=rolling_cost_per_period,
                mode="lines+markers",
                name="rolling Objective Cost",
                marker_color="#FF4162",
                visible = "legendonly"
            )

            layout = Layout(
                title="Plotting Objective Cost Against Time Periods With Ramping Costs",
                xaxis=attr(title="Time Periods", tickangle=-45, tickmode="linear", tick0=1, dtick=1),
                yaxis=attr(title="Objective Cost"),
                showlegend=true
            )

            Plot = plot([trace_cost_per_period_no_ramping, trace_cost_per_period_with_ramping_to_that_period, trace_ramping_cost_per_period, trace_ramping_up_per_period, trace_ramping_down_per_period, trace_rolling_cost_per_period], layout)

            return Plot
        end
    end

##############################################################################################
# Utility Functions
##############################################################################################

    # Function to return the built reference from a data dictionary
    # Useful if we want to look up specific values in the data
    """
        get_ref(data::Dict{String, Any})
    Build and return a reference object from the given power system data dictionary.
    # Arguments
    - `data`: Dictionary containing the power system data for a given case file.
    # Returns
    A reference object containing processed power system data.
    """
    function get_ref(data::Dict{String, Any})
        return PowerModels.build_ref(data)[:it][:pm][:nw][0]
    end
end # module