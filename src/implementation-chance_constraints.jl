function set_model_chance_constraint_objective_function!(power_flow_model::MPOPFModelUncertaintyExtended, factory::DCMPOPFModelFactory)
    model = power_flow_model.model
    data = power_flow_model.data
    T = power_flow_model.time_periods
    ramping_cost = power_flow_model.ramping_cost
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    gen_data = ref[:gen]
    pg = model[:pg]
    ramp_up = model[:ramp_up]
    ramp_down = model[:ramp_down]
    
    @objective(model, Min,
        sum(sum(gen_data[g]["cost"][1]*pg[t,g]^2 + gen_data[g]["cost"][2]*pg[t,g] + gen_data[g]["cost"][3] for g in keys(gen_data)) for t in 1:T) +
        sum(ramping_cost * (ramp_up[t, g] + ramp_down[t, g]) for g in keys(gen_data) for t in 2:T)
    )
end

function set_model_chance_constraint_constraints!(power_flow_model::MPOPFModelUncertaintyExtended, factory::DCMPOPFModelFactory, distributions::Dict, confidence_level::Float64, epsilon::Float64)
    model = power_flow_model.model
    data = power_flow_model.data
    T = power_flow_model.time_periods
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    va = model[:va]
    p = model[:p]
    pg = model[:pg]
    ramp_up = model[:ramp_up]
    ramp_down = model[:ramp_down]
    
    # Calculate z-score for the confidence level
    alpha = 1 - confidence_level
    z = quantile(Normal(0, 1), 1 - alpha/2)  # e.g., for 95% confidence, z = 1.96
    
    p_expr = Dict()
    for t in 1:T
        p_expr[t] = Dict()
    end
    
    # Set up p_expr
    for t in 1:T
        p_expr[t] = Dict([((l, i, j), 1.0 * p[t, (l, i, j)]) for (l, i, j) in ref[:arcs_from]])
        p_expr[t] = merge(p_expr[t], Dict([((l, j, i), -1.0 * p[t, (l, i, j)]) for (l, i, j) in ref[:arcs_from]]))
    end
    
    for t in 1:T
        # Set reference bus angle
        for (i, bus) in ref[:ref_buses]
            @constraint(model, va[t,i] == 0)
        end
        
        # Apply power balance constraints at each bus
        for (b, bus) in ref[:bus]
            bus_loads = ref[:bus_loads][b]
            bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][b]]
            
            # Calculate the left and right quantiles for loads at this bus
            sum_left_quantiles = 0.0
            sum_right_quantiles = 0.0
            mean_load_sum = 0.0
            
            for l in bus_loads
                if haskey(distributions, l)
                    dist = distributions[l]
                    mu = mean(dist)
                    sigma = std(dist)
                    # Calculate left and right quantiles
                    sum_left_quantiles += mu - z * sigma
                    sum_right_quantiles += mu + z * sigma
                    mean_load_sum += mu
                else
                    # If no distribution, use the fixed load value
                    pd_value = ref[:load][l]["pd"]
                    sum_left_quantiles += pd_value
                    sum_right_quantiles += pd_value
                    mean_load_sum += pd_value
                end
            end
            
            # Generation at this bus
            gen_sum = sum(pg[t, g] for g in ref[:bus_gens][b]; init=0.0)
            shunt_sum = sum(shunt["gs"] for shunt in bus_shunts; init=0.0) * 1.0^2
            
            # The constant C is the expected power balance
            C = gen_sum - mean_load_sum - shunt_sum
            
            # Probabilistic constraints
            # P(-epsilon - C <= sum(di) <= epsilon - C) >= 1-alpha
            # Which becomes:
            # -epsilon - C <= sum_left_quantiles
            # epsilon - C >= sum_right_quantiles
            
            @constraint(model, -epsilon - C <= sum_left_quantiles)
            @constraint(model, epsilon - C >= sum_right_quantiles)
            
            # Standard power balance constraint with expected values
            @constraint(model,
                sum(p_expr[t][a] for a in ref[:bus_arcs][b]) ==
                gen_sum - mean_load_sum - shunt_sum
            )
        end
        
        # Branch flow constraints
        for (i,branch) in ref[:branch]
            f_idx = (i, branch["f_bus"], branch["t_bus"])
            
            p_fr = p[t,f_idx]
            
            va_fr = va[t,branch["f_bus"]]
            va_to = va[t,branch["t_bus"]]
            
            g, b = PowerModels.calc_branch_y(branch)
            
            @constraint(model, p_fr == -b*(va_fr - va_to))
            
            @constraint(model, va_fr - va_to <= branch["angmax"])
            @constraint(model, va_fr - va_to >= branch["angmin"])
        end
    end
    
    # Ramping constraints
    for g in keys(ref[:gen])
        for t in 2:T
            @constraint(model, ramp_up[t, g] >= pg[t, g] - pg[t-1, g])
            @constraint(model, ramp_down[t, g] >= pg[t-1, g] - pg[t, g])
        end
    end
end
