############
# Analytic second derivative variable transform.

module HessianReparameterization

using ForwardDiff
using DualNumbers

VERSION < v"0.4.0-dev" && using Docile

export transform_hessian
export get_dx_dy_func, get_d2x_dy2_funcs, get_d2x_dy2, get_df_dy_func, get_d2f_dy2_func

@doc """
Use forward differentiation to get a jacobian from a variable transformation
function.

Args:
	- y_to_x: A function that takes a vector y to a vector x.   Note that it must take
	          generic numeric vectors to numeric vectors. 
	- K: The size of the vectors

Returns:
	- A function that gets the Jacobian of the transform evaluated at y.
""" ->
function get_dx_dy_func(y_to_x::Function, K::Int64)
	# The Jacobians require something that modifies its input in place.
	function y_to_x!(y, x)
		x[:] = y_to_x(y)
	end

	# Calculate the necessary derivatives:
	dx_dy_func_transpose = ForwardDiff.forwarddiff_jacobian(y_to_x!, Float64, fadtype=:dual, n=K, m=K)

	# The default order is transposed relative to the index notation in transform_hessian.
	function dx_dy_func(y)
		dx_dy_func_transpose(y)'
	end
	
	return dx_dy_func
end

function get_d2x_dy2_funcs(y_to_x::Function, K::Int64)
	function y_to_x(y, i::Int64)
		y_to_x(y)[i]
	end
	[ ForwardDiff.forwarddiff_hessian(y -> y_to_x(y, i), Float64, fadtype=:typed, n=K) for i=1:K ]
end

function get_df_dy_func(f::Function, K::Int64)
	ForwardDiff.forwarddiff_gradient(f, Float64, fadtype=:dual, n=K)
end

function get_d2f_dy2_func(f::Function, K::Int64)
	ForwardDiff.forwarddiff_hessian(f, Float64, fadtype=:typed, n=K) 
end

function get_d2x_dy2(d2x_dy2_funcs::Array{Function}, y::Array{Float64, 1})
	K = length(y)
	Float64[ d2x_dy2_funcs[k](y)[i, j] for i=1:K, j=1:K, k=1:K ]
end

function transform_hessian(dx_dy::Array{Float64, 2}, d2x_dy2::Array{Float64, 3},
	                       df_dy::Array{Float64, 1}, d2f_dy2::Array{Float64, 2})
	# Input variables:
	#   dx_dy[i, j] = dx[i] / dy[j]
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

end # module