module ExponentialFamilies

import Distributions
using VariationalModelIndices

VERSION < v"0.4.0-dev" && using Docile

export sparse_mat_from_tuples
export MatrixTuple
export make_ud_index_matrix, linearize_matrix
export unpack_ud_matrix, unpack_ud_trace_coefficients
export get_mvn_parameters_from_derivs, get_mvn_variational_covariance
export wishart_entropy, wishart_e_log_det, get_wishart_variational_covariance
export get_wishart_sufficient_stats_variational_covariance
export get_wishart_parameters_from_derivs

# Multivariate log gamma related functions.
function multivariate_trigamma{T <: Number}(x::T, p::Int64)
	sum([ trigamma(x + 0.5 * (1 - i)) for i=1:p])
end

function multivariate_digamma{T <: Number}(x::T, p::Int64)
	sum([ digamma(x + 0.5 * (1 - i)) for i=1:p])
end

function multivariate_lgamma{T <: Number}(x::T, p::Int64)
	sum([ lgamma(x + 0.5 * (1 - i)) for i=1:p]) +
	p * (p - 1.0) / 4.0 * log(pi)
end

@doc """
Turn a vector of upper diagonal enries back into a symmetric matrix.
""" ->
function unpack_ud_matrix{T <: Number}(ud_vector::Vector{T}; od_scale=1.0)
	k_tot = linearized_matrix_size(length(ud_vector))
	ud_mat = Array(T, (k_tot, k_tot))
	for k1=1:k_tot, k2=1:k_tot
		ud_mat[k1, k2] =
			(k1 <= k2 ? ud_vector[(k1 + (k2 - 1) * k2 / 2)] :
				        ud_vector[(k2 + (k1 - 1) * k1 / 2)])
		ud_mat[k1, k2] *= k1 != k2 ? od_scale: 1.0
	end
	ud_mat
end

@doc """
Convert a vector of upper diagonal entries into a
matrix with halved off-diagonal entries.
This is what's needed to convert the coefficients
of the derivative wrt mu2 into a matrix V such that
tr(V * mu2) = coeffs' * mu2 """ ->
function unpack_ud_trace_coefficients(ud_vector)
	unpack_ud_matrix(ud_vector, od_scale=0.5)
end


# A tuple representing a matrix element
# [row, column, value]
typealias MatrixTuple (Int64, Int64, Float64)

function sparse_mat_from_tuples(tup_array::Array{MatrixTuple})
	sparse(Int64[x[1] for x=tup_array],
		   Int64[x[2] for x=tup_array],
		   Float64[x[3] for x=tup_array])
end


###################################################
# Normal functions

@doc """
beta ~ MVN(beta_mean, beta_cov)
...with elements beta_k

Returns: Cov(beta_k1 beta_k2, beta_k3 beta_k4)
""" ->
function get_mvn_fourth_order_cov(beta_mean::Array{Float64, 1},
	                                beta_cov::Array{Float64, 2},
	                                k1::Int64, k2::Int64, k3::Int64, k4::Int64)
  cov_12 = beta_cov[k1, k2];
  cov_13 = beta_cov[k1, k3];
  cov_14 = beta_cov[k1, k4];
  cov_23 = beta_cov[k2, k3];
  cov_24 = beta_cov[k2, k4];
  cov_34 = beta_cov[k3, k4];

	m_1 = beta_mean[k1];
	m_2 = beta_mean[k2];
	m_3 = beta_mean[k3];
	m_4 = beta_mean[k4];

  return (cov_13 * cov_24 +
  	      cov_14 * cov_23 +
  	      cov_13 * m_2 * m_4 +
  	      cov_14 * m_2 * m_3 +
	        cov_23 * m_1 * m_4 +
	        cov_24 * m_1 * m_3);
end

@doc """
Get the covariance matrix of a multivariate multivariate normal
sufficient statistics given the mean parameters.

E(beta) = beta_mean
Cov(beta) = beta_cov
beta_ind_model = A vector of the model indices of the E(beta)
beta2_ind_model = A matrix containing the model indices of E(beta beta')

Returns:
	An array of MatrixTuple.
""" ->
function get_mvn_variational_covariance(
	beta_mean::Array{Float64, 1}, beta_cov::Array{Float64, 2},
	beta_ind_model::Array{Int64}, beta2_ind_model::Matrix{Int64})

	k_tot = length(beta_ind_model)

	@assert k_tot == length(beta_mean) ==
					size(beta_cov, 1) == size(beta_cov, 2) ==
					size(beta2_ind_model, 1) == size(beta2_ind_model, 2)

	beta_cov_tuples = MatrixTuple[]

	# Get the linear covariances.
	for k1=1:k_tot, k2=1:k_tot
		i1 = beta_ind_model[k1]
		i2 = beta_ind_model[k2]
		push!(beta_cov_tuples, (i1, i2, beta_cov[k1, k2]))
	end

	# Get the covariance between the linear and quadratic terms.
	# This will be cov(mu_k1 mu_k2, mu_i3) := cov(mu2_i12, mu_i3).
	# Avoid double counting since only one mu_i1 mu_i2 is recorded.
	for k1=1:k_tot, k2=1:k1, k3=1:k_tot
		i12 = beta2_ind_model[k1, k2]
		i3 = beta_ind_model[k3]
		this_cov = (beta_mean[k1] * beta_cov[k2, k3] +
		            beta_mean[k2] * beta_cov[k1, k3])
		push!(beta_cov_tuples, (i3, i12, this_cov))
		push!(beta_cov_tuples, (i12, i3, this_cov))
	end

	# Get the covariance between the quadratic terms.
	# This will be cov(mu_k1 mu_k2, mu_k3 mu_k4) := cov(mu2_i12, mu_i34).
	# Avoid double counting since only one mu_k1 mu_k2
	# and mu_k3 mu_k4 is recorded.
	for k1=1:k_tot, k2=1:k1, k3=1:k_tot, k4=1:k3
		i12 = beta2_ind_model[k1, k2]
		i34 = beta2_ind_model[k3, k4]
		this_cov = get_mvn_fourth_order_cov(beta_mean, beta_cov,
			k1, k2, k3, k4)

		push!(beta_cov_tuples, (i12, i34, this_cov))
	end

	beta_cov_tuples
end


@doc """
Given a log likelihood's derivatives with respect to a MVN's mean
parameters, return its mean and covariance.  The log likelihood is

beta_deriv' beta + beta2_deriv' beta2

TODO: document beta2 a little more clearly.

Args:
	- beta_deriv: The derivative of the log likelihood with respect to beta
	- beta2_deriv: The derivative of the log likelihood with respect to
		           the upper diagonal of beta beta'.

Returns:
	- E(beta), Cov(beta)
""" ->
function get_mvn_parameters_from_derivs(
		beta_deriv::Array{Float64}, beta2_deriv::Array{Float64})
	k_tot = length(beta_deriv)
	@assert length(beta2_deriv) == k_tot * (k_tot + 1) / 2
	beta_dist =
		Distributions.MvNormalCanon(
			beta_deriv, -2 * unpack_ud_trace_coefficients(beta2_deriv))
	mean(beta_dist), cov(beta_dist)
end

@doc """
Get the normal covariance for a scalar normal with expectation
beta_mean, expecation of the square beta2_mean, and in columns
beta_ind_model and beta2_ind_model respectively.
""" ->
function get_normal_variational_covariance(
		beta_mean, beta2_mean, beta_ind_model, beta2_ind_model)

	norm_cov = MatrixTuple[]

	norm_var = beta2_mean - beta_mean ^ 2
	@assert norm_var >= 0.0

	# Get the linear term variance
	push!(norm_cov, (beta_ind_model, beta_ind_model, norm_var))

	# Get the covariance between the linear and quadratic terms.
	this_cov = 2 * beta_mean * norm_var
	push!(norm_cov, (beta_ind_model, beta2_ind_model, this_cov))
	push!(norm_cov, (beta2_ind_model, beta_ind_model, this_cov))

	# Get the covariance between the quadratic terms.
	this_cov = 2 * norm_var ^ 2 + 4 * norm_var * (beta_mean ^ 2)
	push!(norm_cov, (beta2_ind_model, beta2_ind_model, this_cov))

	norm_cov
end


###################################################
# Wishart functions

@doc """
The covariance of wishart distributed random variables.
""" ->
function get_wishart_variational_covariance(
		v0::Matrix{Float64}, wn::Float64, ud_ind::Matrix{Int64})
	@assert size(v0, 1) == size(v0, 2)
	k_tot = size(v0, 1)
	k_ud = int(k_tot * (k_tot + 1) / 2)
	w_cov = Array(Float64, k_ud, k_ud)

	for j1=1:k_tot, i1=1:j1, j2=1:k_tot, i2=1:j2
		ind_1 = ud_ind[j1, i1]
		ind_2 = ud_ind[j2, i2]
		w_cov[ind_1, ind_2] = wn * (v0[i1, j2] * v0[i2, j1] + v0[i1, i2] * v0[j1, j2])
		if ind_1 != ind_2
			w_cov[ind_2, ind_1] = w_cov[ind_1, ind_2]
		end
	end

	w_cov
end

@doc """
The variance of the log determininant of a wishart matrix.
""" ->
function get_wishart_log_det_variance(wn::Float64, k_tot::Int64)
	multivariate_trigamma(float(wn) / 2, k_tot)
end

@doc """
The covariance of the log determininant of a wishart matrix and wishart variables.
""" ->
function get_wishart_cross_variance(v0::Matrix{Float64}, ud_ind::Matrix{Int64})
	2.0 * linearize_matrix(v0, ud_ind)
end

@doc """
The covariance matrix of the sufficient statistics of a wishart distribution.
""" ->
function get_wishart_sufficient_stats_variational_covariance(
		v0::Matrix{Float64}, wn::Float64, lambda_i, log_det_lambda_i, ud_ind)
	@assert size(v0, 1) == size(v0, 2) == size(ud_ind, 1) == size(ud_ind, 2)
	k_tot = size(v0, 1)
	k_ud = k_tot * (k_tot + 1) / 2
	@assert length(lambda_i) == k_ud

	cov_triplets = MatrixTuple[]

	log_det_cov = get_wishart_log_det_variance(wn, k_tot)
	cross_cov = get_wishart_cross_variance(v0, ud_ind)
	lambda_cov = get_wishart_variational_covariance(v0, wn, ud_ind)

	push!(cov_triplets,
		  (log_det_lambda_i, log_det_lambda_i, log_det_cov))

	for i=1:length(lambda_i)
		push!(cov_triplets,
			  (log_det_lambda_i, lambda_i[i], cross_cov[i]))
  		push!(cov_triplets,
			  (lambda_i[i], log_det_lambda_i, cross_cov[i]))
		for j=1:length(lambda_i)
			push!(cov_triplets,
				  (lambda_i[i], lambda_i[j], lambda_cov[i, j]))
		end
	end

	cov_triplets
end


@doc """
For a Wishart distribution with mean wn v0_inv^{-1},
evaluate the expected log determinant.
""" ->
function wishart_e_log_det(wn::Float64, v0_inv::Matrix{Float64})
	@assert size(v0_inv, 1) == size(v0_inv, 2)
	p = size(v0_inv, 1)
	multivariate_digamma(0.5 * wn, p) + p * log(2) - logdet(v0_inv)
end

@doc """
For a Wishart distribution with mean wn v0_inv^{-1},
evaluate the entropy.
""" ->
function wishart_entropy(wn::Float64, v0_inv::Matrix{Float64}, k_tot::Int64)
	0.5 * k_tot * (k_tot + 1) * log(2) +
	-0.5 * logdet(v0_inv) * (k_tot + 1.0) +
	multivariate_lgamma(0.5 * wn, k_tot) -
	0.5 * (wn - k_tot - 1.0) * multivariate_digamma(0.5 * wn, k_tot) +
	0.5 * wn * k_tot
end


function get_wishart_parameters_from_derivs(
		lambda_deriv::Matrix{Float64}, log_det_lambda_deriv::Float64)

	@assert size(lambda_deriv, 1) == size(lambda_deriv, 2)
	k_tot = size(lambda_deriv, 1)
	wn = 2. * log_det_lambda_deriv + 1. + k_tot
	v0 = -0.5 * inv(lambda_deriv)
	v0_inv = -2.0 * lambda_deriv

	wn, v0, v0_inv
end


###################################################
# Gamma distribution functions

function get_gamma_parameters_from_derivs(
		tau_deriv::Float64, log_tau_deriv::Float64)

	tau_alpha = log_tau_deriv + 1
	tau_beta = -tau_deriv

	@assert tau_alpha >= 0
	@assert tau_beta >= 0

	tau_alpha, tau_beta
end

@doc """
Covariance of sufficient statistics for a gamma distribution with
mean tau_alpha / tau_beta.
""" ->
function get_gamma_variational_covariance(tau_alpha::Float64, tau_beta::Float64,
	                                        e_tau_col::Int64, e_log_tau_col::Int64)
	tau_cov = MatrixTuple[]
	push!(tau_cov, (e_tau_col,     e_tau_col,     tau_alpha / (tau_beta ^ 2)))
	push!(tau_cov, (e_log_tau_col, e_log_tau_col, trigamma(tau_alpha)))
	push!(tau_cov, (e_tau_col,     e_log_tau_col, 1 / tau_beta))
	push!(tau_cov, (e_log_tau_col, e_tau_col,     1 / tau_beta))
	tau_cov
end


function gamma_entropy(tau_alpha::Float64, tau_beta::Float64)
	tau_alpha - log(tau_beta) + lgamma(tau_alpha) +
		(1 - tau_alpha) * digamma(tau_alpha)
end



end # module
