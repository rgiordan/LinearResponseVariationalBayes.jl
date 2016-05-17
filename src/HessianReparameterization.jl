############
# Analytic second derivative variable transform.

module HessianReparameterization

using ForwardDiff
using DualNumbers
using VariationalModelIndices

VERSION < v"0.4.0-dev" && using Docile

export transform_hessian, get_moment_hess
export get_dx_dy_func, get_d2x_dy2_funcs, get_d2x_dy2
export get_df_dy_func, get_d2f_dy2_func

@doc """
Use forward differentiation to get a jacobian from a variable transformation
function.

Args:
	- y_to_x: A function that takes a vector y to a vector x.
	          Note that it must map generic numeric vectors to numeric vectors.
	- K: The size of the vectors

Returns:
	- A function that gets the Jacobian of the transform evaluated at y.
""" ->
function get_dx_dy_func(y_to_x::Function, K::Int64)

	# Calculate the necessary derivatives:
	dx_dy_func_transpose = ForwardDiff.jacobian(y_to_x, mutates=false)

	# The default order is transposed relative to the index notation in
	# transform_hessian.
	function dx_dy_func{T <: Number}(y::Array{T})
		dx_dy_func_transpose(y)'
	end

	return dx_dy_func
end

function get_d2x_dy2_funcs(y_to_x::Function, K::Int64)
	function y_to_x{T <: Number}(y::Array{T}, i::Int64)
		y_to_x(y)[i]
	end
	[ ForwardDiff.hessian(y -> y_to_x(y, i), mutates=false) for i=1:K ]
end

function get_df_dy_func(f::Function, K::Int64)
	ForwardDiff.gradient(f, mutates=false)
end

function get_d2f_dy2_func(f::Function, K::Int64)
	ForwardDiff.hessian(f, mutates=false)
end

function get_d2x_dy2(d2x_dy2_funcs::Array{Function}, y::Array{Float64, 1})
	K = length(y)
	result = Float64[ d2x_dy2_funcs[k](y)[i, j] for i=1:K, j=1:K, k=1:K ]
	result
end

function transform_hessian(dx_dy::Array{Float64, 2}, d2x_dy2::Array{Float64, 3},
	                         df_dy::Array{Float64, 1}, d2f_dy2::Array{Float64, 2})
	# Input variables:
	#   dx_dy[i, j] = dx[i] / dy[j] # TODO: it actually requires -- and is being
	#                                 passsed -- dx_dy[i, j]= dx[j] / dy[i]
	#   d2x_dy2[i, j, k] = d2 x[k] / (dy[i] dy[j])
	#   df_dy[i] = df / dy[i]
	#   d2f_dy2[i, j] = d2f / (dy[i] dy[j])
	#
	# Returns:
	#   d2f / dx2

	K = size(dx_dy, 1)
	@assert K == size(dx_dy, 2) == size(d2x_dy2, 1) == size(d2x_dy2, 2) ==
	        size(df_dy, 1) == size(d2f_dy2, 1) == size(d2f_dy2, 2)

	# The same as dx_dy \ d2f_dy2 * inv(dx_dy').  This is the only term
	# if you are at an optimum since in that case df_dy = 0.
	opt_term = dx_dy \ (dx_dy \ d2f_dy2)'

	df_dx = dx_dy \ df_dy

	# This is dJ_dx[i, j, k] = dJ[i, j] / dx[k] = d / dx[k] (dx[j] / dy[i])
	dJ_dx = Float64[ (dx_dy \ slice(d2x_dy2, :, i, j)[:])[k] for i=1:K, j=1:K, k=1:K ]

	j_term = Array(Float64, K, K)
	for j=1:K
		j_term[:, j] = -(dx_dy \ slice(dJ_dx, :, :, j)[:, :]) * df_dx
	end

	j_term + opt_term
end


@doc """
Given a model parameter hessian, moment transform, and indices, compute the
hessian with respect to moment parameters.
""" ->
function get_moment_hess(model_params::Vector{Float64},
	                       model_grad::Vector{Float64}, model_hess::Matrix{Float64},
	                       indices::ModelIndices, transform::Function)
	# Get the necessary derivatives to do a change of variables in the Hessian.
	# A legend for the variable names to ease legibility:
 	# x = Moment parameters
 	# y = Model parameters
 	# 0 = Untransformed parameters
 	# 1 = Transformed parameters
 	# 2 = Constant parameters

	transformed_param_keys = keys(indices.trans_input);
	untransformed_param_keys = setdiff(keys(indices.model_params), transformed_param_keys);

	# The model indices of the transformed parameters in the correct order.
	transformed_indices = zeros(Int64, indices.num_transformed_indices)
	for trans_param in transformed_param_keys
		transformed_indices[indices.trans_input[trans_param]] =
			indices.model_params[trans_param]
	end

	trans_model_params = model_params[transformed_indices];
 	trans_moment_params =
		VariationalModelIndices.get_transformed_params(model_params, indices, transform)

	dx_dy_func = get_dx_dy_func(transform, indices.num_transformed_indices);
	d2x_dy2_funcs = get_d2x_dy2_funcs(transform, indices.num_transformed_indices);

	dx_dy = dx_dy_func(trans_model_params);

	d2x_dy2 = get_d2x_dy2(d2x_dy2_funcs, trans_model_params);

	# Rename for clarity.
	full_df_dy = model_grad;
	full_d2f_dy2 = model_hess;

	df_dy_1 = full_df_dy[transformed_indices];

	d2f_dy2_11 = full_d2f_dy2[transformed_indices, transformed_indices];
	d2f_dy2_1A = full_d2f_dy2[transformed_indices, :]; # A for "all"

	# Compute the submatrix of the transformed Hessian corresponding to
	# the transformed parameters.
	d2f_dx2_11 = transform_hessian(dx_dy, d2x_dy2, df_dy_1, d2f_dy2_11);

	# Compute the submatrix of the transformed Hessian corresponding to
	# the mixed partials between the transformed and untransformed parameters.
	# Note that the columns are indexed in model order, where the rows
	# are indexed by indices.trans_output.
	d2f_dx2_1A = dx_dy \ d2f_dy2_1A;
	d2f_dx2_A1 = d2f_dx2_1A';

	# Populate the hessian with respect to the moment parameters.
	moment_hess =
		zeros(Float64, indices.num_moment_indices, indices.num_moment_indices);
	for param0_row in untransformed_param_keys
		param0_row_indices = indices.model_params[param0_row]
		moment0_row_indices = indices.moment_params[param0_row]

		# The untransformed by untransformed block
		for param0_col in untransformed_param_keys
			param0_col_indices = indices.model_params[param0_col]
			moment0_col_indices = indices.moment_params[param0_col]
			moment_hess[moment0_row_indices, moment0_col_indices] =
				full_d2f_dy2[param0_row_indices, param0_col_indices]
		end

		# The untransformed by transformed blocks
		for param1 in keys(indices.trans_output)
			param1_indices = indices.trans_output[param1]
			moment1_indices = indices.moment_params[param1]
			moment_hess[moment0_row_indices, moment1_indices] =
				d2f_dx2_A1[param0_row_indices, param1_indices]
			moment_hess[moment1_indices, moment0_row_indices] =
				d2f_dx2_1A[param1_indices, param0_row_indices]
		end
	end

	# The transformed by transformed block
	for param1_row in keys(indices.trans_output)
		param1_row_indices = indices.trans_output[param1_row]
		moment1_row_indices = indices.moment_params[param1_row]
		for param1_col in keys(indices.trans_output)
			param1_col_indices = indices.trans_output[param1_col]
			moment1_col_indices = indices.moment_params[param1_col]
			moment_hess[moment1_row_indices, moment1_col_indices] =
				d2f_dx2_11[param1_row_indices, param1_col_indices]
		end
	end

	# Finally, get the submatrices corresponding to the "constant" parameters
	# (the priors and the data).
	const_param_hess =
		zeros(Float64, indices.num_const_indices, indices.num_moment_indices);
	const_params = keys(indices.model_const)
	@assert collect(keys(indices.model_const)) == collect(keys(indices.moment_const))

	for param2 in const_params
		param2_indices = indices.model_const[param2]
		moment2_indices = indices.moment_const[param2]
		for param0 in untransformed_param_keys
			param0_indices = indices.model_params[param0]
			moment0_indices = indices.moment_params[param0]
			const_param_hess[moment2_indices, moment0_indices] =
				full_d2f_dy2[param2_indices, param0_indices]
		end
		for param1 in keys(indices.trans_output)
			param1_indices = indices.trans_output[param1]
			moment1_indices = indices.moment_params[param1]
			const_param_hess[moment2_indices, moment1_indices] =
				d2f_dx2_A1[param2_indices, param1_indices]
		end
	end

	# The model indices of the constant parameters in the correct order.
	const_indices = zeros(Int64, indices.num_const_indices)
	for const_param in const_params
		const_indices[indices.moment_const[const_param]] = indices.model_const[const_param]
	end
	const_const_hess = full_d2f_dy2[const_indices, const_indices];

	moment_hess, const_param_hess, const_const_hess,
	dx_dy, d2x_dy2, full_df_dy, full_d2f_dy2, transformed_indices
end


end # module
