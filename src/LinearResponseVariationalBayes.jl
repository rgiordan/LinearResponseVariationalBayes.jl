# To use, you must clone directly from github:
# https://github.com/rgiordan/LinearResponseVariationalBayes.jl

module LinearResponseVariationalBayes

require(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "VariationalModelIndices.jl"))
require(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "ExponentialFamilies.jl"))
require(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "SubModels.jl"))
require(joinpath(Pkg.dir("LinearResponseVariationalBayes"), "src", "HessianReparameterization.jl"))

import VariationalModelIndices
import ExponentialFamilies
import SubModels
import HessianReparameterization

end # module
