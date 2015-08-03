module SubModels

using JuMP
VERSION < v"0.4.0-dev" && using Docile

import MathProgBase
import ReverseDiffSparse
import Optim


export JuMPObjective
export get_objective_val, get_objective_deriv!, get_objective_hess!
export optimize_subobjective

@doc """
A type for holding a particular JuMP sub-objective.
""" ->
type JuMPObjective
	name::ASCIIString
	m::JuMP.Model
	m_eval::JuMP.JuMPNLPEvaluator

	# Which variables have nonzero derivatives.
	vars::Array{Bool, 1}

	colval::Array{Float64}

	# Stuff for the hessian
	hess_struct::(Array{Int64,1}, Array{Int64,1})
	hess_vec::Array{Float64,1}
	numconstr::Int64
	n_params::Int64

	# An in-place gradient
	grad::Array{Float64, 1}

end

JuMPObjective(m::Model) = JuMPObjective(m, "model", Bool[true for i in 1:length(m.colVal)])

JuMPObjective(m::Model, name::ASCIIString, vars::Array{Bool, 1}) = begin
	colval = m.colVal

	m_const_mat = JuMP.prepConstrMatrix(m);
	m_eval = JuMP.JuMPNLPEvaluator(m, m_const_mat);
	MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])

	# Structures for the hessian.
	hess_struct = MathProgBase.hesslag_structure(m_eval);
	hess_vec = zeros(length(hess_struct[1]));
	numconstr = (length(m_eval.m.linconstr) +
		         length(m_eval.m.quadconstr) +
		         length(m_eval.m.nlpdata.nlconstr))

	grad = zeros(length(m.colVal))

	n_params = sum(vars)

	JuMPObjective(name, m, m_eval, vars, colval, hess_struct, hess_vec, numconstr, n_params, grad)
end

JuMPObjective(m::Model, m_eval::JuMP.JuMPNLPEvaluator, name::ASCIIString, vars::Array{Bool, 1}) = begin
	colval = m.colVal

	m_const_mat = JuMP.prepConstrMatrix(m);

	# Structures for the hessian.
	hess_struct = MathProgBase.hesslag_structure(m_eval);
	hess_vec = zeros(length(hess_struct[1]));
	numconstr = (length(m_eval.m.linconstr) +
		         length(m_eval.m.quadconstr) +
		         length(m_eval.m.nlpdata.nlconstr))

	grad = zeros(length(m.colVal))

	n_params = sum(vars)

	JuMPObjective(name, m, m_eval, vars, colval, hess_struct, hess_vec, numconstr, n_params, grad)
end


@doc """
z_par should be of length sum(jo.vars).
""" ->
function get_objective_val(z_par::Array{Float64, 1}, jo::JuMPObjective; verbose=false)
	@assert length(z_par) == jo.n_params
	jo.colval[jo.vars] = z_par
	elbo_val = jo.m_eval.eval_f_nl(jo.colval)
	if verbose
		println("$(jo.name) elbo: $elbo_val")
	end
	elbo_val
end

@doc """
Calculates the derivative with respect to all variables, even excluded ones.
""" ->
function get_full_objective_deriv!(z_par::Array{Float64, 1}, jo::JuMPObjective)
	@assert length(z_par) == jo.n_params
	jo.colval[jo.vars] = z_par
	MathProgBase.eval_grad_f(jo.m_eval, jo.grad, jo.colval)
	jo.grad
end

@doc """
Calculates the derivative only with respect to changeable variables.
""" ->
function get_objective_deriv!(z_par::Array{Float64, 1}, jo::JuMPObjective, grad::Array{Float64, 1})
	get_full_objective_deriv!(z_par, jo)
	grad[:] = jo.grad[jo.vars]
end

@doc """
Returns the hessian with respect to all variables.
""" ->
function get_full_objective_hess!(z_par::Array{Float64, 1}, jo::JuMPObjective)
	@assert length(z_par) == jo.n_params
	jo.colval[jo.vars] = z_par
	MathProgBase.eval_hesslag(jo.m_eval, jo.hess_vec,
		                      jo.colval, 1.0, zeros(jo.numconstr))
	this_hess_ld = sparse(jo.hess_struct[1], jo.hess_struct[2],
		                  jo.hess_vec, length(jo.colval), length(jo.colval))
	this_hess = full(this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld))))
	this_hess[jo.vars, jo.vars]
end

@doc """
Returns the hessian only with respect to changeable variables.
""" ->
function get_objective_hess!(z_par::Array{Float64, 1}, jo::JuMPObjective, hess)
	hess[:,:] = get_full_objective_hess!(z_par, jo)
	hess
end

@doc """
From an initial value of z_init for all the variables in m, optimize
the variables associated with the sub-objective jo.
""" ->
function optimize_subobjective(z_init::Array{Float64, 1}, jo::JuMPObjective;
	                           scale=1.0, hess_reg=0.0, show_trace=false, iterations=5)
	@assert length(z_init) == length(jo.colval)
	z_par = z_init[jo.vars]
	jo.colval = z_init
	function get_local_objective_val(z_par)
		scale * get_objective_val(z_par, jo)
	end
	function get_local_objective_deriv!(z_par, grad)
		get_objective_deriv!(z_par, jo, grad)
		grad[:] = grad * scale
	end
	function get_local_objective_hess!(z_par, hess)
		if hess_reg < Inf
			get_objective_hess!(z_par, jo, hess)
			hess[:, :] = (hess + eye(length(z_par)) * hess_reg) * scale
		else
			hess[:, :] = scale * eye(length(z_par))
		end
	end
	optim_res = Optim.optimize(get_local_objective_val,
		                         get_local_objective_deriv!,
		                         get_local_objective_hess!,
		                         z_par, method=:newton, show_trace=show_trace, iterations=iterations)
	z_final = z_init
	z_final[jo.vars] = optim_res.minimum
	z_final, optim_res.f_minimum, optim_res
end


end # module
