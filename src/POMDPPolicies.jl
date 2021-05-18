module POMDPPolicies

using LinearAlgebra
using Random
using StatsBase # for Weights
using SparseArrays # for sparse vectors in alpha_vector.jl
using Parameters
using Distributions # For logpdf extenstion in playback policy
using FiniteHorizonPOMDPs

using POMDPs
import POMDPs: action, value, solve, updater

using BeliefUpdaters
using POMDPModelTools

using Base.Iterators # for take

"""
    actionvalues(p::Policy, s)

returns the values of each action at state s in a vector
"""
function actionvalues end

export
    actionvalues

export
    AlphaVectorPolicy,
    alphavectors,
    alphapairs

include("alpha_vector.jl")

export
    StagedPolicy

include("staged_policy.jl")

export
    FunctionPolicy,
    FunctionSolver

include("function.jl")

export
    RandomPolicy,
    RandomSolver

include("random.jl")

export
    VectorPolicy,
    VectorSolver,
    ValuePolicy

include("vector.jl")

export
    StochasticPolicy,
    UniformRandomPolicy,
    CategoricalTabularPolicy

include("stochastic.jl")

export LinearDecaySchedule,
       EpsGreedyPolicy,
       SoftmaxPolicy,
       ExplorationPolicy,
       loginfo

include("exploration_policies.jl")

export
    PolicyWrapper,
    payload

include("utility_wrapper.jl")

export
    showpolicy

include("pretty_printing.jl")

export
    PlaybackPolicy

include("playback.jl")

end
