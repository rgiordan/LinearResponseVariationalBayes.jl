# To use, you must clone directly from github:
# https://github.com/rgiordan/JuMPVariationalBayes.jl

module JuMPVariationalBayes

require(joinpath(Pkg.dir("JuMPVariationalBayes"), "src", "ExponentialFamilies.jl"))
require(joinpath(Pkg.dir("JuMPVariationalBayes"), "src", "SubModels.jl"))
require(joinpath(Pkg.dir("JuMPVariationalBayes"), "src", "HessianReparameterization.jl"))

import ExponentialFamilies
import SubModels
import HessianReparameterization

end # module
