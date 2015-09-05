# Objects for keeping track of indices in models.
module VariationalModelIndices

VERSION < v"0.4.0-dev" && using Docile

export make_ud_index_matrix, linearize_matrix, linearized_matrix_size
export IndexDict, ModelIndices
export get_transformed_params, get_moment_params

@doc """
Return a matrix of indices that can be used to index into a linearization
of the upper diagonal of a matrix.
""" ->
function make_ud_index_matrix(k::Int64)
	ud_mat = Array(Int64, (k, k))
	for k1=1:k, k2=1:k
		ud_mat[k1, k2] =
			(k1 <= k2 ? (k1 + (k2 - 1) * k2 / 2) :
				        (k2 + (k1 - 1) * k1 / 2))
	end
	ud_mat
end

@doc """
Convert the upper diagonal of a matrix to a vector using the indieces of ud_ind.
""" ->
function linearize_matrix(mat::Matrix{Float64}, ud_ind::Matrix{Int64})
	k_ud = maximum(ud_ind)
	k_tot = size(ud_ind, 1)

	mat_lin = Array(Float64, k_ud)
	for k1=1:k_tot, k2=1:k1
		mat_lin[ud_ind[k1, k2]] = mat[k1, k2]
	end
	mat_lin
end

@doc """
Get a matrix size from the number of upper triangular elements.
""" ->
function linearized_matrix_size(k_ud::Int64)
	k_tot = 0.5 * (sqrt(1 + 8 * k_ud) - 1)
	k_tot_int = int(k_tot)
	@assert int(k_tot) == k_tot
	k_tot_int
end



# A dictionary from (symbol, extra information) to a vector of indices.
typealias IndexDict Dict{Any, Vector{Int64}}

@doc """
Index dictionaries that characterize a VB solution.

model_params: The parameters in the optimization problem
moment_params: The moment parameters of the variational solution
model_const: Constant parameters (e.g. prior parameters) in the model
moment_const: Constant parameters (e.g. prior parameters) in the LRVB matrix
trans_input: Inputs of transformed parameters (must be in model_params)
trans_output: Outputs of transformed parameteres (must be in moment_params)
""" ->
type ModelIndices
	# Indices into the objective (JuMP) model
	model_params::IndexDict

	# Indices into the vector of moment parameters.
	moment_params::IndexDict

	# Indices for constants (e.g. prior parameters) in the model
	# and moment matrix, respectively.
	model_const::IndexDict
	moment_const::IndexDict

	# Model parameters that are transformed into moment parameters.
	trans_input::IndexDict
	trans_output::IndexDict

	num_moment_indices::Int64
	num_const_indices::Int64
	num_transformed_indices::Int64
end


ModelIndices(model_params::IndexDict, moment_params::IndexDict,
             model_const::IndexDict, moment_const::IndexDict,
             trans_input::IndexDict, trans_output::IndexDict) = begin

  for key in keys(trans_input)
    @assert haskey(model_params, key)  "Missing key $(key) from model_params"
  end

  for key in keys(trans_output)
    @assert haskey(moment_params, key) "Missing key $(key) from moment_params"
  end

  num_moment_indices = maximum([ maximum(val) for val in values(moment_params) ])
  num_const_indices = maximum([ maximum(val) for val in values(moment_const) ])
  num_transformed_indices =
    sum(Int64[ length(indices) for (param, indices) in trans_output ])

  ModelIndices(model_params, moment_params,
	             model_const, moment_const, trans_input, trans_output,
               num_moment_indices, num_const_indices, num_transformed_indices)
end


function get_transformed_params(model_params::Vector{Float64},
                                indices::ModelIndices, transform::Function)
  # Get the transformed parameter vector.
  trans_input_params = zeros(Float64, indices.num_transformed_indices)
  for trans_param in keys(indices.trans_input)
  	trans_input_params[indices.trans_input[trans_param]] =
  		model_params[indices.model_params[trans_param]]
  end
  transform(trans_input_params)
end


function get_moment_params(model_params::Vector{Float64},
                           indices::ModelIndices, transform::Function)

	transformed_params = keys(indices.trans_input);
	untransformed_params = setdiff(keys(indices.model_params), transformed_params);

  trans_moment_params = get_transformed_params(model_params, indices, transform)

	moment_params = zeros(Float64, indices.num_moment_indices);
	for param0 in untransformed_params
		param0_indices = indices.model_params[param0]
		moment0_indices = indices.moment_params[param0]
		moment_params[moment0_indices] = model_params[param0_indices]
	end
	for param1 in keys(indices.trans_output)
		param1_indices = indices.trans_output[param1]
		moment1_indices = indices.moment_params[param1]
		moment_params[moment1_indices] = trans_moment_params[param1_indices]
	end

	moment_params
end


end # Module
