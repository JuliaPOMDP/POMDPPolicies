let
    T = 50
    N = 50

    pomdp = BabyPOMDP()

    rsum = 0.0
    policy = solve(BlindPolicySolver(verbose=true), pomdp)
    for i in 1:N
        sim = RolloutSimulator(max_steps=T, rng=MersenneTwister(i))
        rsum += simulate(sim, pomdp, policy)
    end

    @show rsum/N
end