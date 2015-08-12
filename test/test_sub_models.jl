
using Base.Test
using JuMPVariationalBayes.SubModels
using JuMP

println("Testing SubModels.")


function test_sub_models()
  m = Model()
  @defVar(m, x)
  @defVar(m, y)
  @defVar(m, z)

  setValue(x, 1.0)
  setValue(y, 2.0)
  setValue(z, 3.0)

  @defNLExpr(obj, x * y + z ^ 2)
  @setNLObjective(m, Min, obj)

  getValue(obj)

  jo_full = JuMPObjective(m)

  z_col = z.col
  sub_vars = Bool[ false for i=1:length(m.colVal)]
  sub_vars[z_col] = true
  jo_sub = JuMPObjective(m, "sub", sub_vars)

  param_val = Float64[5, 6, 7]
  @test_approx_eq(get_objective_val(param_val, jo_full),
                  get_objective_val([param_val[z_col]], jo_sub))
  @test_approx_eq(get_objective_val(param_val, jo_full), getValue(obj))
  SubModels.get_full_objective_deriv!(param_val, jo_full)
  SubModels.get_full_objective_deriv!([param_val[z_col]], jo_sub)

  @test_approx_eq(jo_full.grad, jo_sub.grad)

  sub_grad = Float64[0.0]
  SubModels.get_objective_deriv!([param_val[z_col]], jo_sub, sub_grad)
  @test_approx_eq(jo_full.grad[z_col], sub_grad)

  @test_approx_eq(
    SubModels.get_objective_hess(param_val, jo_full)[z_col, z_col],
    SubModels.get_objective_hess([param_val[z_col]], jo_sub))

  sub_hess = zeros(1, 1)
  full_hess = zeros(3, 3)

  SubModels.get_objective_hess!(param_val, jo_full, full_hess)
  SubModels.get_objective_hess!([param_val[z_col]], jo_sub, sub_hess)
  @test_approx_eq(full_hess[z_col, z_col], sub_hess)

  optimize_subobjective(param_val, jo_sub)
  @test_approx_eq(jo_sub.colval[z_col], 0.0)
end

test_sub_models()
