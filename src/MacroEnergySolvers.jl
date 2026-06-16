module MacroEnergySolvers

    using JuMP
    using Distributed
    using DistributedArrays
    using Pkg
    using Dates, Logging
    using LinearAlgebra, Random

    include("benders/planning.jl")
    include("benders/subproblems.jl")
    include("benders/regularization.jl")
    include("benders/algorithms.jl")
    include("logging.jl")
    include("benders/mga.jl")   # added for MGA

    export benders
    export benders_mga          # added for MGA 

end