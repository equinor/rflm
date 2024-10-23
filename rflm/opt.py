import numpy as np

# General function to adjust the objective_function to allow for fixed parameters. 
def optimize_wrapper(free_params, fixed_params, fixed_indices, objective_function, *args, **kwargs):
    # Create a full parameter array
    full_params = np.zeros(len(free_params) + len(fixed_params))
    free_index = 0
    fixed_index = 0
    for i in range(len(full_params)):
        if i in fixed_indices:
            full_params[i] = fixed_params[fixed_index]
            fixed_index += 1
        else:
            full_params[i] = free_params[free_index]
            free_index += 1
    return objective_function(full_params, *args, **kwargs) 

# General function to adjust the constraints to allow for fixed parameters
def adjust_constraints(constraints, fixed_indices, fixed_params):
    def adjusted_constraint(constraint, fixed_indices, fixed_params):
        def new_constraint(free_params):
            full_params = np.zeros(len(free_params) + len(fixed_params))
            free_index = 0
            fixed_index = 0
            for i in range(len(full_params)):
                if i in fixed_indices:
                    full_params[i] = fixed_params[fixed_index]
                    fixed_index += 1
                else:
                    full_params[i] = free_params[free_index]
                    free_index += 1
            return constraint(full_params)
        return new_constraint

    new_constraints = []
    for constraint in constraints:
        new_constraints.append({
            'type': constraint['type'],
            'fun': adjusted_constraint(constraint['fun'], fixed_indices, fixed_params)
        })
    return new_constraints