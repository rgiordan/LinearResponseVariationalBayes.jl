# To use, you must clone directly from github:
# https://github.com/rgiordan/LinearResponseVariationalBayes.jl

module LinearResponseVariationalBayes

push!(LOAD_PATH, joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src"))

# include(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "VariationalModelIndices.jl"))
# include(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "ExponentialFamilies.jl"))
# include(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "SubModels.jl"))
# include(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "HessianReparameterization.jl"))

import VariationalModelIndices
import ExponentialFamilies
import SubModels
import HessianReparameterization

end # module
