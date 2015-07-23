
using JuMPVariationalBayes
using HessianReparameterization
using Base.Test

function test_hessian_reparameterization()
	# An example function and transform.
	K = 2
	a = 1.5

	function f(y) 
		y[1]^3 + y[2]^2 + a * y[1] * y[2]
	end

	function y_to_x(y)
		[ y[1] * y[2],
		  y[1] + y[2]]
	end

	dx_dy_func = get_dx_dy_func(y_to_x, K)
	d2x_dy2_funcs = get_d2x_dy2_funcs(y_to_x, K)
	df_dy_func = get_df_dy_func(f, K)
	d2f_dy2_func = get_d2f_dy2_func(f, K)

	y = Float64[1.5, 3]

	df_dy = df_dy_func(y)
	d2f_dy2 = d2f_dy2_func(y)
	dx_dy = dx_dy_func(y)
	d2x_dy2 = get_d2x_dy2(d2x_dy2_funcs, y)

	d2f_dx2 = transform_hessian(dx_dy, d2x_dy2, df_dy, d2f_dy2)

	# Test with numeric derivatives.
	function df_dx_func(y)
		dx_dy_func(y) \ df_dy_func(y)
	end

	eps = 1e-6
	y_a = y + eps * Float64[1, 0]
	y_b = y + eps * Float64[0, 1]

	x = y_to_x(y)
	x_a = y_to_x(y_a) - x
	x_b = y_to_x(y_b) - x

	f_a = f(y_a) - f(y)
	f_b = f(y_b) - f(y)

	# Test the first derivative.
	df_dx_est = vcat(x_a', x_b') \ Float64[f_a, f_b]
	df_dx_func(y)

	@test_approx_eq_eps df_dx_est df_dx_func(y) 1e-5

	# Test the second derivative.
	g_a = df_dx_func(y_a) - df_dx_func(y)
	g_b = df_dx_func(y_b) - df_dx_func(y)

	d2f_dx2_est =  vcat(x_a', x_b') \ vcat(g_a', g_b')
	@test_approx_eq_eps d2f_dx2_est d2f_dx2 1e-4
end


function test_lgamma_hessian_reparameterization()
	# An example function and transform to check the lgamma and digamma functions.

	K = 2
	a = 1.5

	function f(y) 
		y[1]^3 + y[2]^2 + a * y[1] * y[2]
	end

	function y_to_x(y)
		[ lgamma(0.1 * y[1] + 0.9 * y[2]),
		  digamma(0.9 * y[1] + 0.1 * y[2]) ]
	end

	dx_dy_func = get_dx_dy_func(y_to_x, K)
	d2x_dy2_funcs = get_d2x_dy2_funcs(y_to_x, K)
	df_dy_func = get_df_dy_func(f, K)
	d2f_dy2_func = get_d2f_dy2_func(f, K)

	y = Float64[1.2, 2.1]

	df_dy = df_dy_func(y)
	d2f_dy2 = d2f_dy2_func(y)
	dx_dy = dx_dy_func(y)
	d2x_dy2 = get_d2x_dy2(d2x_dy2_funcs, y)

	d2f_dx2 = transform_hessian(dx_dy, d2x_dy2, df_dy, d2f_dy2)

	# Test with numeric derivatives.
	function df_dx_func(y)
		dx_dy_func(y) \ df_dy_func(y)
	end

	eps = 1e-6
	y_a = y + eps * Float64[1, 0]
	y_b = y + eps * Float64[0, 1]

	x = y_to_x(y)
	x_a = y_to_x(y_a) - x
	x_b = y_to_x(y_b) - x

	f_a = f(y_a) - f(y)
	f_b = f(y_b) - f(y)

	# Test the first derivative.
	df_dx_est = vcat(x_a', x_b') \ Float64[f_a, f_b]
	df_dx_func(y)

	@test_approx_eq_eps df_dx_est df_dx_func(y) 1e-5

	# Test the second derivative.
	g_a = df_dx_func(y_a) - df_dx_func(y)
	g_b = df_dx_func(y_b) - df_dx_func(y)

	d2f_dx2_est =  vcat(x_a', x_b') \ vcat(g_a', g_b')
	@test_approx_eq_eps d2f_dx2_est d2f_dx2 1e-4
end


test_hessian_reparameterization()
test_lgamma_hessian_reparameterization()