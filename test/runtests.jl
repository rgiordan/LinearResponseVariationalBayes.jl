using JuMPVariationalBayes
using Base.Test

include(joinpath(Pkg.dir("JuMPVariationalBayes"), "test", "test_hessian_reparam.jl"))
include(joinpath(Pkg.dir("JuMPVariationalBayes"), "test", "test_exponential_families.jl"))
