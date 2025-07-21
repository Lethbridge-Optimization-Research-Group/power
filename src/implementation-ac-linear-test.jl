using Distributions

function set_model_variables!(power_flow_model::AbstractMPOPFModel, factory::LinTMPOPFModelFactory)
    model = power_flow_model.model
    T = power_flow_model.time_periods
    ref = PowerModels.build_ref(power_flow_model.data)[:it][:pm][:nw][0]
    bus_data = ref[:bus]
    gen_data = ref[:gen]
    branch_data = ref[:branch]
    
    @variable(model, va[t in 1:T, i in keys(bus_data)])
    @variable(model, bus_data[i]["vmin"] <= vm[t in 1:T, i in keys(bus_data)] <= bus_data[i]["vmax"], start=1.0)
    @variable(model, gen_data[i]["pmin"] <= pg[t in 1:T, i in keys(gen_data)] <= gen_data[i]["pmax"])
    @variable(model, gen_data[i]["qmin"] <= qg[t in 1:T, i in keys(gen_data)] <= gen_data[i]["qmax"])
    @variable(model, -branch_data[l]["rate_a"] <= p[t in 1:T, (l,i,j) in ref[:arcs]] <= branch_data[l]["rate_a"])
    @variable(model, -branch_data[l]["rate_a"] <= q[t in 1:T, (l,i,j) in ref[:arcs]] <= branch_data[l]["rate_a"])
    @variable(model, ramp_up[t in 2:T, g in keys(gen_data)] >= 0)
    @variable(model, ramp_down[t in 2:T, g in keys(gen_data)] >= 0)
end

function set_model_objective_function!(power_flow_model::AbstractMPOPFModel, factory::LinTMPOPFModelFactory)
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

function set_model_constraints!(power_flow_model::AbstractMPOPFModel, factory::LinTMPOPFModelFactory)
    model = power_flow_model.model
    data = power_flow_model.data
    T = power_flow_model.time_periods
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    gen_data = ref[:gen]
    load_data = ref[:load]

    va = model[:va]
    p = model[:p]
    q = model[:q]
    pg = model[:pg]
    qg = model[:qg]
    vm = model[:vm]
	ramp_up = model[:ramp_up]
    ramp_down = model[:ramp_down]

    factors = power_flow_model.factors
    
	# power balance constraints for time periods
    for t in 1:T

		# 
        for (i, bus) in ref[:ref_buses]
            @constraint(model, va[t,i] == 0)
        end

		# 
        for (i, bus) in ref[:bus]
            #d = sampling("Normal")
            d = 1
            bus_loads = [load_data[l] for l in ref[:bus_loads][i]]
            bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

            @constraint(model,
                sum(p[t,a] for a in ref[:bus_arcs][i]) ==
                sum(pg[t, g] for g in ref[:bus_gens][i]) -
                sum(load["pd"] * d* factors[t] for load in bus_loads) -
                sum(shunt["gs"] for shunt in bus_shunts)*vm[t,i]^2
            )

            @constraint(model,
                sum(q[t,a] for a in ref[:bus_arcs][i]) ==
                sum(qg[t, g] for g in ref[:bus_gens][i]) -
                sum(load["qd"] * d * factors[t] for load in bus_loads) +
                sum(shunt["bs"] for shunt in bus_shunts)*vm[t,i]^2
            )
        end

		# active ( p ) and reactive ( q ) power constraints
        for (i, branch) in ref[:branch]
            f_idx = (i, branch["f_bus"], branch["t_bus"])
            t_idx = (i, branch["t_bus"], branch["f_bus"])

            p_to = p[t,t_idx]
            q_to = q[t,t_idx]
            p_fr = p[t,f_idx]
            q_fr = q[t,f_idx]

            va_fr = va[t,branch["f_bus"]]
            va_to = va[t,branch["t_bus"]]

            vm_fr = vm[t,branch["f_bus"]]
            vm_to = vm[t,branch["t_bus"]]

            g, b = PowerModels.calc_branch_y(branch)
            tr, ti = PowerModels.calc_branch_t(branch)
            ttm = tr^2 + ti^2

            g_fr = branch["g_fr"]
            b_fr = branch["b_fr"]
            g_to = branch["g_to"]
            b_to = branch["b_to"]

            @constraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(0.5042084743962969*vm_fr^2+0.5032217155898757*vm_to^2-0.022411308142930787*va_fr+0.007495337723430631*va_to-0.012301401193273165) + (-b*tr-g*ti)/ttm*(-0.00172351897934318*vm_fr^2+0.005607659368881494*vm_to^2+1.0241445591684628*va_fr-1.025189439104993*va_to-0.004433921484692217) )
            @constraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(0.5042084743962969*vm_fr^2+0.5032217155898757*vm_to^2-0.022411308142930787*va_fr+0.007495337723430631*va_to-0.012301401193273165) + (-b*tr-g*ti)/ttm*(-0.00172351897934318*vm_fr^2+0.005607659368881494*vm_to^2+1.0241445591684628*va_fr-1.025189439104993*va_to-0.004433921484692217))
    
            # To side of the branch flow
            @constraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(0.5042084743962969*vm_fr^2+0.5032217155898757*vm_to^2-0.022411308142930787*va_fr+0.007495337723430631*va_to-0.012301401193273165) + (-b*tr+g*ti)/ttm*(+0.00172351897934318*vm_fr^2-0.005607659368881494*vm_to^2-1.0241445591684628*va_fr+1.025189439104993*va_to+0.004433921484692217))
            @constraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(0.5042084743962969*vm_fr^2+0.5032217155898757*vm_to^2-0.022411308142930787*va_fr+0.007495337723430631*va_to-0.012301401193273165) + (-g*tr-b*ti)/ttm*(+0.00172351897934318*vm_fr^2-0.005607659368881494*vm_to^2-1.0241445591684628*va_fr+1.025189439104993*va_to+0.004433921484692217))
            
            @constraint(model, va_fr - va_to <= branch["angmax"])
            @constraint(model, va_fr - va_to >= branch["angmin"])

            @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
            @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
        end
    end

	# ramp up and ramp down constraints
    for g in keys(gen_data)
        for t in 2:T
            @constraint(model, ramp_up[t, g] >= pg[t, g] - pg[t-1, g])
            @constraint(model, ramp_down[t, g] >= pg[t-1, g] - pg[t, g])
        end
    end
end

function sampling(i::String)
    d = nothing
    if i == "Uniform"
        d = rand(Uniform(.9,1.1))
    else
        mu = 1
        sigma = 0.01
        d = rand(Normal(mu, sigma))
    end
    return d
end