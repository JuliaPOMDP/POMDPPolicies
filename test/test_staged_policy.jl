import Base: ==, hash

# Define struct not yet exported in POMDPs.jl framework
struct AlphaVec
    alpha::Vector{Float64} # alpha vector
    action::Any # action associated wtih alpha vector
end

AlphaVec() = AlphaVec([0.0], 0)

# define alpha vector equality
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))
==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)


import BeliefUpdaters: pdf
pdf(b::DiscreteBelief, s) = b.b[typeof(HorizonLength(b.pomdp)) == InfiniteHorizon ?
                                stateindex(b.pomdp, s) : stage_stateindex(b.pomdp, s)]

import BeliefUpdaters: initialize_belief
function initialize_belief(bu::DiscreteUpdater, dist::FiniteHorizonPOMDPs.InStageDistribution)
    state_list = ordered_stage_states(bu.pomdp, stage(dist))
    ns = length(state_list)
    b = zeros(ns)
    belief = DiscreteBelief(bu.pomdp, state_list, b)
    for s in support(dist)
        sidx = stage_stateindex(bu.pomdp, s)
        belief.b[sidx] = pdf(dist, s)
    end
    return belief
end

import BeliefUpdaters: update
function update(bu::DiscreteUpdater, b::DiscreteBelief, a, o)
    pomdp = bu.pomdp
    state_space = b.state_list
    bp = zeros(length(state_space))

    for (si, s) in enumerate(state_space)
        si = stage_stateindex(pomdp, s)

        if pdf(b, s) > 0.0
            td = transition(pomdp, s, a)
            for (sp, tp) in weighted_iterator(td)
                spi = stage_stateindex(pomdp, sp)
                op = obs_weight(pomdp, s, a, sp, o) # shortcut for observation probability from POMDPModelTools
                op2 = pdf(observation(pomdp, a, sp), o)

                bp[spi] += op * tp * b.b[si]
            end
        end
    end

    bp_sum = sum(bp)

    if bp_sum == 0.0
        error("""
              Failed discrete belief update: new probabilities sum to zero.

              b = $b
              a = $a
              o = $o

              Failed discrete belief update: new probabilities sum to zero.
              """)
    end

    # Normalize
    bp ./= bp_sum

    return DiscreteBelief(pomdp, ordered_stage_states(pomdp, stage(pomdp, o)+1), bp)
end

methods(update)

#test
let
    hor = 5
    pomdp = fixhorizon(TigerPOMDP(), hor)

    bu = DiscreteUpdater(pomdp.m)
    b0 = initialize_belief(bu, initialstate_distribution(pomdp.m))

    Γs = []
    # I only know correct alpha vectors from horizon stage
    for i in 1:4
        push!(Γs, [AlphaVec([0., 0.], 0), AlphaVec([0., 0.], 0), AlphaVec([0., 0.], 0)])
    end
    push!(Γs, [AlphaVec([10., 0.], 1), AlphaVec([0., 10.], 2), AlphaVec([-0.5, -0.5], 0)])


    staged_policies = [AlphaVectorPolicy(pomdp.m, length(stage_states(pomdp, t)), [a.alpha for a in Γs[t]],
        convert(Vector{actiontype(pomdp)}, [a.action for a in Γs[t]])) for t in 1:horizon(pomdp)]

    policy = StagedPolicy(pomdp, staged_policies)

    # these values were gotten from FiVI.jl tests
    @test Set(alphapairs(policy, hor)) == Set([[10., 0.]=>1, [0., 10.]=>2, [-0.5, -0.5]=>0])
    @test Set(alphavectors(policy, hor)) == Set([[10., 0.], [0., 10.], [-0.5, -0.5]])

    # initial belief is 50% tiger position confidence
    @test isapprox(value(policy, b0, hor), hor)
    @test isapprox(value(policy, [0.5, 0.5], hor), 5)
    # I am not sure whether the result should be [-0.5, 5. 5.] or [-1., 5. 5.]
    # may check the correctness in the future
    @test_broken isapprox(actionvalues(policy, b0, hor), [-1, 5., 5.])
    @test length(actionvalues(policy, b0, hor)) == length(actions(pomdp))

    # The agent only has one decision, so he has to risk choosing one door
    @test action(policy, b0, hor) == 1 || action(policy, b0, hor) == 2

    # try pushing new vector
    push!(policy, [0.0,0.0], 0, hor)
    @test isapprox(length(policy.staged_policies[hor].alphas), 4)

    # test simulation
    up = updater(policy)
    @test isapprox(simulate(RolloutSimulator(), pomdp, policy, up), 4.435, rtol=.1)

    # test infinite horizon pomdp
    pomdp = BabyPOMDP()
    @test_throws ArgumentError StagedPolicy(pomdp, staged_policies)
end
