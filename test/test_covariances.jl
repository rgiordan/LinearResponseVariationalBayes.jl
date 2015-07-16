using Covariances


#TODO: get this working

function test_wishart()

	# Check the wishart things
	p = 2
	mu_sigma = eye(p) + repmat([0.1], p, p)
	lambda = inv(mu_sigma)

	wn = 10.
	v0 = inv(lambda)

	w = Wishart(wn, v0)

	# Looks good
	mean([ logdet(rand(w)) for i=1:10000 ])
	wishart_e_log_det(wn, inv(v0))

	mean([ rand(w) for i=1:10000 ])
	v0 * wn

	wn = 10.
	v0 = inv(lambda)

	w = Wishart(wn, v0)

	# You actually need a lot more to test this well but it's slow.
	w_test_n = 10000
	w_lin = reduce(vcat, [ linearize_matrix(rand(w), vb.ud_ind)'
		                    for i=1:w_test_n ]);
	cov(w_lin)

	get_wishart_variational_covariance(vb.k_tot, v0, wn, vb.ud_ind)
end
