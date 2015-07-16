
function make_ud_index_matrix(k)
	ud_mat = Array(Int64, (k, k))
	for k1=1:k, k2=1:k
		ud_mat[k1, k2] =
			(k1 <= k2 ? (k1 + (k2 - 1) * k2 / 2) :
				        (k2 + (k1 - 1) * k1 / 2))
	end
	ud_mat
end
