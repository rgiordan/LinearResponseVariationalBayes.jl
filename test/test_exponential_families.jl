
using Base.Test
using JuMPVariationalBayes.ExponentialFamilies
import Distributions

println("Testing ExponentialFamilies.")


function rel_diff(x, y)
	(x - y) ./ y
end


function test_wishart()
	srand(42)
	p = 2
	p_ud = int(p * (p + 1) / 2)
	mu_sigma = eye(p) + repmat([0.2], p, p)
	lambda = inv(mu_sigma)
	n_draws = int(5e5)

	wn = 100.
	v0 = inv(lambda)

	w = Distributions.Wishart(wn, v0)
	w_draws = [ rand(w) for i=1:n_draws ];
	w_logdet_draws = Float64[ logdet(w_draw) for w_draw in w_draws ];

	# Test the means.
	@test_approx_eq_eps(rel_diff(mean(w_logdet_draws), wishart_e_log_det(wn, inv(v0))),
	                    0.0, 1e-3)

	@test_approx_eq_eps(rel_diff(mean(w_draws), (v0 * wn)), zeros(p, p), 1e-2)

	@test_approx_eq(ExponentialFamilies.wishart_entropy(wn, lambda, p),
	                Distributions.entropy(w))

	# Test the covariances.
	ud_ind = make_ud_index_matrix(p)
	w_lin = Array(Float64, n_draws, p_ud)
	for row = 1:n_draws
		w_lin[row, :] = linearize_matrix(w_draws[row], ud_ind)
	end

	wishart_cov =
		ExponentialFamilies.get_wishart_variational_covariance(v0, wn, ud_ind)
	@test_approx_eq_eps(rel_diff(cov(w_lin), wishart_cov), zeros(p_ud, p_ud), 5e-2)

	wishart_suff_stats = hcat(w_lin, w_logdet_draws);
	wishart_suff_stats_sample_cov = cov(wishart_suff_stats)
	wishart_cross_sample_cov = wishart_suff_stats_sample_cov[p_ud + 1,:][1:p_ud]
	wishart_cross_cov = ExponentialFamilies.get_wishart_cross_variance(v0, ud_ind)
	@test_approx_eq_eps(rel_diff(wishart_cross_sample_cov, wishart_cross_cov),
	                    zeros(p_ud), 1e-1)

	wishart_log_det_var = ExponentialFamilies.get_wishart_log_det_variance(wn, p)
	@test_approx_eq_eps(rel_diff(var(w_logdet_draws), wishart_log_det_var),
											0.0, 5e-2)

	wishart_suff_stats_cov = full(sparse_mat_from_tuples(
		get_wishart_sufficient_stats_variational_covariance(v0, wn, collect(1:p_ud),
			p_ud + 1, ud_ind)))

	@test_approx_eq_eps(rel_diff(wishart_suff_stats_sample_cov, wishart_suff_stats_cov),
	                    zeros(p_ud + 1, p_ud + 1), 1e-1)
end


function test_normal()
	srand(42)
	p = 2
	mu_mean = convert(Array{Float64}, collect(1:p))
	mu_sigma = 0.5 * (eye(p) + repmat([0.2], p, p))

	mu_dist = Distributions.MvNormal(mu_mean, mu_sigma)
	n_draws = int(1e6)
	mu_draws = rand(mu_dist, n_draws)'
	second_order_sample_cov =
		Float64[ cov(mu_draws[:, k1] .* mu_draws[:, k2],
		             mu_draws[:, k3] .* mu_draws[:, k4])
		         for k1=1:p, k2=1:p, k3=1:p, k4=1:p ]
	second_order_cov =
 		Float64[ ExponentialFamilies.get_mvn_fourth_order_cov(mu_mean, mu_sigma,
		                                                      k1, k2, k3, k4)
						 for k1=1:p, k2=1:p, k3=1:p, k4=1:p ]
	@test_approx_eq_eps(rel_diff(second_order_sample_cov, second_order_cov),
	                    zeros(p, p, p, p), 5e-2)

	suff_stat_draws = hcat(mu_draws,
	                       mu_draws[:, 1] .* mu_draws[:, 1],
												 mu_draws[:, 1] .* mu_draws[:, 2],
												 mu_draws[:, 2] .* mu_draws[:, 2])

  suff_stat_cov = full(sparse_mat_from_tuples(
		ExponentialFamilies.get_mvn_variational_covariance(
			mu_mean, mu_sigma, collect(1:2), Int64[ 3 4; 4 5])))
	suff_stat_sample_cov = cov(suff_stat_draws)
	@test_approx_eq_eps(rel_diff(suff_stat_sample_cov, suff_stat_cov),
											zeros(5, 5), 5e-2)

	suff_stat_draws_1d = hcat(mu_draws[:, 1], mu_draws[:, 1] .^ 2)
	suff_stat_1d_cov = full(sparse_mat_from_tuples(
		ExponentialFamilies.get_normal_variational_covariance(
			mu_mean[1], mu_mean[1] ^ 2 + mu_sigma[1, 1], 1, 2)))
	suff_stat_1d_sample_cov = cov(suff_stat_draws_1d)
	@test_approx_eq_eps(rel_diff(suff_stat_1d_sample_cov, suff_stat_1d_cov),
											zeros(2, 2), 5e-2)

end


function test_gamma()
	alpha = 5.0
	beta = 10.0
	gamma_dist = Distributions.Gamma(alpha, 1 / beta)

	n_draws = int(1e5)
	gamma_draws = rand(gamma_dist, n_draws)
	gamma_suff_draws = hcat(gamma_draws, log(gamma_draws))

	gamma_suff_cov = full(sparse_mat_from_tuples(
		ExponentialFamilies.get_gamma_variational_covariance(alpha, beta, 1, 2)))
	gamma_suff_sample_cov = cov(gamma_suff_draws)

	@test_approx_eq_eps(rel_diff(gamma_suff_sample_cov, gamma_suff_cov), zeros(2, 2), 1e-2)
	@test_approx_eq(ExponentialFamilies.gamma_entropy(alpha, beta),
	                Distributions.entropy(gamma_dist))
end

test_wishart()
test_normal()
test_gamma()
