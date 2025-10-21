# Chance Constraints Notes

- This is a short description of how the chance constraints code works and was programmed for optimizing OPF under uncertainty.

## The Model

```
- Branch: Probabilistic-OPF
- File location: src/Probabilistic-OPF/probabilistic-OPF.jl
- Function: create_probabilistic_model
```

- This function creates the model needed for solving chance constraints OPF. 
- It parses data, creates a power flow model, sets variables, sets the objective function, and sets the constraints.
- Setting the variables function: `set_model_variables!`
- Setting the Objective function function: `set_probabilistic_objective_function!`
- Setting the constraints function: `set_probabilistic_constraints!`
- It returns a power flow model with our specified characteristics.

## The Chance Constraints

```
- Branch: Probabilistic-OPF
- File location: src/Probabilistic-OPF/probabilistic-OPF.jl
- Function: set_probabilistic_constraints!
```

- This function is responsible for setting the chance constraints.
- All the constraints are identical to DC OPF constraints except the following: (lines 94-126)

```jl
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
```

- The most important thing is thing to realize is that we are passing in a distribution to the function. That distribution is accessed and used for each load.
- C is the expected power balance: `C = gen_sum - mean_load_sum - shunt_sum`
- We want to code this: `P(-epsilon - C <= sum(di) <= epsilon - C) >= 1-alpha`
- The last two lines of the above code snippet is the result of converting the probability function into code.

## Creating and optimizing a Model

```
- Branch: Probabilistic-OPF
- File location: src/Probabilistic-OPF/probabilistic-OPF.jl
- Function: run_probabilistic_method
```

- This function is a way that we can run and optimize one of these models.
- The inputs are the case file path, confidence level, epsilon, variation, type and variation value.
- The distribution variable is a dictionary mapped to each load.
- This is an example of how to get the distributions.
- Note at this time it is **normally distributed**.
- Make sure to set the file path to the correct case file.

```jl
using MPOPF

file_path = "././Cases/case14.m"
variation_type = :relative
variation_value = 0.15

distributions = setup_demand_distributions(file_path, variation_type, variation_value)
```

- Here is an example output:

```
Dict{Any, Any} with 11 entries:
  5  => Distributions.Normal{Float64}(μ=0.112, σ=0.0168)
  8  => Distributions.Normal{Float64}(μ=0.035, σ=0.00525)
  1  => Distributions.Normal{Float64}(μ=0.217, σ=0.03255)
  6  => Distributions.Normal{Float64}(μ=0.295, σ=0.04425)
  11 => Distributions.Normal{Float64}(μ=0.149, σ=0.02235)
  9  => Distributions.Normal{Float64}(μ=0.061, σ=0.00915)
  3  => Distributions.Normal{Float64}(μ=0.478, σ=0.0717)
  7  => Distributions.Normal{Float64}(μ=0.09, σ=0.0135)
  4  => Distributions.Normal{Float64}(μ=0.076, σ=0.0114)
  2  => Distributions.Normal{Float64}(μ=0.942, σ=0.1413)
  10 => Distributions.Normal{Float64}(μ=0.135, σ=0.02025)
```

- If interested in how the distribution function works it can be located here: `src/implementation_uncertainty.jl`. In this file we can also find the core for the old scenario based approach.
- After the `run_probabilistic_method` function is called it will return an optimized model which can be queried for values such as the objective function and pg values.
- Here is example of how to run and query the returned model:

```jl
# Run the probabilistic method on the case14.m file
probabilistic_model = run_probabilistic_method("././Cases/case14.m", confidence_level=0.95, epsilon=5.0, variation_type=:relative, variation_value=0.15)

pg_vals = JuMP.value.(probabilistic_model.model[:pg])
obj_val = objective_value(probabilistic_model.model)
```


## Saving Results

```
- Branch: Probabilistic-OPF
- File location: src/Probabilistic-OPF/saving-results.jl
- Function: run_epsilon_sweep
```

- This function is an example of how to run the `run_probabilistic_method` function systematically for different value of epsilon and save the results to a **.csv** file.
- The function that actually runs the sweep is `run_parameter_sweep` but it is complicated and it is best to look at the example.
- We can run sweeps for epsilon, confidence level, and variation value.
- All those functions are in the same file: `src/Probabilistic-OPF/saving-results.jl`
- The results will be saved in a specific directory structure:

```
├── probabilistic-results
│   ├── case118
│   │   ├── confidence_level
│   │   │   └── results.csv
│   │   ├── epsilon
│   │   │   └── results.csv
│   │   └── variation_value
│   │       └── results.csv
│   ├── case14
│   │   ├── confidence_level
│   │   │   └── results.csv
│   │   ├── epsilon
│   │   │   └── results.csv
│   │   └── variation_value
│   │       └── results.csv
```

- This structure is very important since it is what the graphing functions expect to be able to graph our results.
- Each .csv file will contain the sweep results of that run.

## Graphing Results

```
- Branch: Probabilistic-OPF
- File location: src/Probabilistic-OPF/plot-probabilistic-results.jl
- Function: plot_all_results
```

- This file contains a couple of functions that can be used to plot the results that are saved in the directory structure described above.
- The `plot_all_results` function specifically will plot every .csv file it finds.
- We can isolate a specific case and sweep to plot using the `plot_case_sweep_results` function. Here is an Example:

```jl
# Example usage:
# Plot results for a specific case and sweep
plot_case_sweep_results("case14", "epsilon")

# Or we can simply plot everything and compare the plots
plot_all_results()
```

- The plots will be saved in a similar directory structure since they can then be easily displayed by some html code. This is so that the code knows where to expect each file.
- Here is what the directory structure will look like:

```
├── probabilistic-plots
│   ├── case118
│   │   ├── confidence_level
│   │   │   ├── generator_outputs.html
│   │   │   ├── objective_function.html
│   │   │   └── total_generation.html
│   │   ├── epsilon
│   │   │   ├── generator_outputs.html
│   │   │   ├── objective_function.html
│   │   │   └── total_generation.html
│   │   └── variation_value
│   │       ├── generator_outputs.html
│   │       ├── objective_function.html
│   │       └── total_generation.html
│   ├── case14
│   │   ├── confidence_level
│   │   │   ├── generator_outputs.html
│   │   │   ├── objective_function.html
│   │   │   └── total_generation.html
│   │   ├── epsilon
│   │   │   ├── generator_outputs.html
│   │   │   ├── objective_function.html
│   │   │   └── total_generation.html
│   │   └── variation_value
│   │       ├── generator_outputs.html
│   │       ├── objective_function.html
│   │       └── total_generation.html
```

- Each html file is it's own graph.

## For the Future

- If the goal is to include different correlations for loads then we simply need to modify the constraints in the model.
- If we want to pass in different distributions we would simply generate them outside the `run_probabilistic_method` function and pass it as a parameter.
- If we want to extend to AC OPF we would need to  write the necessary constraints.
- Currently I have been running test with GUROBI however it might be interesting to try multiple solvers.



