
using Base.Test
using JuMPVariationalBayes.ExponentialFamilies
import Distributions

function test_wishart()
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
	@test_approx_eq_eps(mean(w_logdet_draws) ./ wishart_e_log_det(wn, inv(v0)),
	                    1.0, 1e-3)

	@test_approx_eq_eps(mean(w_draws) ./ (v0 * wn), ones(p, p), 1e-2)

	@test_approx_eq(ExponentialFamilies.wishart_entropy(wn, lambda, p),
	                Distributions.entropy(w))

	# Test the covariances.
	ud_ind = make_ud_index_matrix(p)
	w_lin = Array(Float64, n_draws, p_ud)
	for row = 1:n_draws
		w_lin[row, :] = linearize_matrix(w_draws[row], ud_ind)
	end

	wishart_cov = ExponentialFamilies.get_wishart_variational_covariance(v0, wn, ud_ind)
	@test_approx_eq_eps((cov(w_lin) - wishart_cov) ./ wishart_cov,
	                    zeros(p_ud, p_ud), 5e-2)

	wishart_suff_stats = hcat(w_lin, w_logdet_draws);
	wishart_suff_stats_sample_cov = cov(wishart_suff_stats)
	wishart_cross_cov = wishart_suff_stats_sample_cov[p_ud + 1,:][1:p_ud]
	@test_approx_eq_eps(wishart_cross_cov ./ ExponentialFamilies.get_wishart_cross_variance(v0, ud_ind),
	                    ones(p_ud), 5e-2)

	@test_approx_eq_eps(var(w_logdet_draws) / ExponentialFamilies.get_wishart_log_det_variance(wn, p),
											1.0, 1e-2)

	wishart_suff_stats_cov = full(sparse_mat_from_tuples(
		get_wishart_sufficient_stats_variational_covariance(v0, wn, collect(1:p_ud), p_ud + 1, ud_ind)))

	@test_approx_eq_eps(wishart_suff_stats_sample_cov ./ wishart_suff_stats_cov,
	                    ones(p_ud + 1, p_ud + 1), 1e-1)
end
