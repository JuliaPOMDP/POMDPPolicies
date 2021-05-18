
"""
StagedPolicy

Policy for finite horizon POMDP solvers.
Wraps multiple Policies, one for each stage.
"""
struct StagedPolicy{M<:Union{POMDP,MDP}, P<:Policy}<:Policy
    pomdp::M
    staged_policies::Array{P, 1}

    function StagedPolicy(m::M, p::Vector{P}) where {M<:Union{POMDP,MDP}, P<:Policy}
        HorizonLength(m) == InfiniteHorizon() ? throw(ArgumentError("Argument pomdp should be
                             valid Finite Horizon POMDP with methods from FiniteHorizonPOMDPs.jl
                             interface implemented. If you are completely sure that you implemented
                             all of them, you should also check if you have defined
                             HorizonLength(::Type{<:MyFHMDP})")) : new{M, P}(m, p)
     end
end

### functional methods ###
value(p::StagedPolicy, b, t) = t <= horizon(p.pomdp) ? value(p.staged_policies[t], b) : 0.

action(p::StagedPolicy, b, t) = t <= horizon(p.pomdp) ? action(p.staged_policies[t], b) : ordered_actions(p.pomdp)[1]

# Given that, for example, POMDPSimulators.jl does not work with action(p, b, t),
# we need to somehow introduce the stage variable
function action(p::StagedPolicy, b::DiscreteBelief)
    t = -1
    for i in 1:length(b.b)
        if b.b[i] > 0.0
            if t < 0
                t = stage(p.pomdp, b.state_list[i])
            elseif t != stage(p.pomdp, b.state_list[i])
                error("Not all states in the belief had the same stage")
            end
        end
    end
    return action(p, b, t)
end

updater(p::StagedPolicy) = DiscreteUpdater(p.pomdp)

"""
Return an iterator of alpha vector-action pairs of given stage in the policy.
"""
alphapairs(p::StagedPolicy, t::Int64) = alphapairs(p.staged_policies[t])

"""
Return the alpha vectors of given stage
"""
alphavectors(p::StagedPolicy, t::Int64) = alphavectors(p.staged_policies[t])

actionvalues(p::StagedPolicy, b, t::Int64) = actionvalues(p.staged_policies[t], b)

function Base.push!(p::StagedPolicy{<:Union{POMDP,MDP},<:Policy}, alpha::Vector{Float64}, a, t)
    push!(p.staged_policies[t].alphas, alpha)
    push!(p.staged_policies[t].action_map, a)
end
