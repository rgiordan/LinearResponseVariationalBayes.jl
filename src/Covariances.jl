#module Covariances

import Distributions

VERSION < v"0.4.0-dev" && using Docile

export sparse_mat_from_tuples
export MatrixTuple

# A tuple representing a matrix element
# [row, column, value]
typealias MatrixTuple (Int64, Int64, Float64)

function sparse_mat_from_tuples(tup_array::Array{MatrixTuple})
	sparse(Int64[x[1] for x=tup_array],
		   Int64[x[2] for x=tup_array],
		   Float64[x[3] for x=tup_array])
end

@doc """
beta ~ MVN(beta_mean, beta_cov)
...with elements beta_k

Returns: Cov(beta_k1 beta_k2, beta_k3 beta_k4)
""" ->
function get_mvn_fourth_order_cov(beta_mean::Array{Float64, 1}, beta_cov::Array{Float64, 2},
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

E(beta) = v_beta
Cov(beta) = v_beta_cov
beta_ind_model = A vector of the model indices of the E(beta)
beta2_ind_model = A matrix containing the model indices of E(beta beta')

Returns:
	An array of MatrixTuple.
""" ->
function get_mvn_variational_covariance(v_beta::Array{Float64, 1}, v_beta_cov::Array{Float64, 2},
	beta_ind_model::Array{Int64}, beta2_ind_model::Array{Int64})

	k_tot = length(beta_ind_model)

	@assert k_tot == length(v_beta) == size(v_beta_cov, 1) == size(v_beta_cov, 2) ==
		size(beta2_ind_model, 1) == size(beta2_ind_model, 2)

	beta_cov = MatrixTuple[]

	# Get the linear covariances.
	for k1=1:k_tot, k2=1:k_tot
		i1 = beta_ind_model[k1]
		i2 = beta_ind_model[k2]
		push!(beta_cov, (i1, i2, v_beta_cov[k1, k2]))
	end

	# Get the covariance between the linear and quadratic terms.
	# This will be cov(mu_k1 mu_k2, mu_i3) := cov(mu2_i12, mu_i3).
	# Avoid double counting since only one mu_i1 mu_i2 is recorded.
	for k1=1:k_tot, k2=1:k1, k3=1:k_tot
		i12 = beta2_ind_model[k1, k2]
		i3 = beta_ind_model[k3]
		this_cov = v_beta[k1] * v_beta_cov[k2, k3] + v_beta[k2] * v_beta_cov[k1, k3]
		push!(beta_cov, (i3, i12, this_cov))
		push!(beta_cov, (i12, i3, this_cov))
	end

	# Get the covariance between the quadratic terms.
	# This will be cov(mu_k1 mu_k2, mu_k3 mu_k4) := cov(mu2_i12, mu_i34).
	# Avoid double counting since only one mu_k1 mu_k2
	# and mu_k3 mu_k4 is recorded.
	for k1=1:vb_reg.k_tot, k2=1:k1, k3=1:vb_reg.k_tot, k4=1:k3
		i12 = beta2_ind_model[k1, k2]
		i34 = beta2_ind_model[k3, k4]
		this_cov = get_mvn_fourth_order_cov(v_beta, v_beta_cov,
			k1, k2, k3, k4)

		push!(beta_cov, (i12, i34, this_cov))
	end

	beta_cov
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
function get_mvn_parameters_from_derivs(beta_deriv::Array{Float64}, beta2_deriv::Array{Float64})
	k_tot = length(beta_deriv)
	@assert length(beta2_deriv) == k_tot * (k_tot - 1) / 2

	function unpack_ud_matrix(ud_vector, k_tot)
		# Convert a vector of upper diagonal entries into a
		# matrix with halved off-diagonal entries.
		# This is what's needed to convert the coefficients
		# of the derivative wrt beta2 into a matrix V such that
		# tr(V * beta2) = coeffs' * beta2

		ud_mat = Array(Float64, (k_tot, k_tot))
		for k1=1:k_tot, k2=1:k_tot
			ud_mat[k1, k2] =
				(k1 <= k2 ? ud_vector[(k1 + (k2 - 1) * k2 / 2)] :
					        ud_vector[(k2 + (k1 - 1) * k1 / 2)])
			ud_mat[k1, k2] *= k1 != k2 ? 0.5: 1.
		end
		ud_mat
	end

	beta_dist = Distributions.MvNormalCanon(beta_deriv, -2 * unpack_ud_matrix(beta2_deriv))
	mean(beta_dist), cov(beta_dist)
end

@doc """
Get the normal covariance for a scalar normal with expectation
e_norm, expecation of the square e_norm2, and in columns
e_col and e2_col respectively.
""" ->
function get_normal_variational_covariance(e_norm, e_norm2, e_col, e2_col)

	norm_cov = MatrixTuple[]

	norm_var = e_norm2 - e_norm ^ 2
	# Get the linear term variance
	push!(norm_cov, (e_col, e_col, norm_var))

	# Get the covariance between the linear and quadratic terms.
	this_cov = 2 * e_norm * norm_var
	push!(norm_cov, (e_col, e2_col, this_cov))
	push!(norm_cov, (e2_col, e_col, this_cov))			

	# Get the covariance between the quadratic terms.
	this_cov = 2 * norm_var ^ 2 + 4 * norm_var * (e_norm ^ 2)
	push!(norm_cov, (e2_col, e2_col, this_cov))

	norm_cov
end




#end # module