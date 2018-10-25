let
    pomdp = BabyPOMDP()

    bu = DiscreteUpdater(pomdp)
    b0 = initialize_belief(bu, initialstate_distribution(pomdp))

    # these values were gotten from FIB.jl
    # alphas = [-29.4557 -36.5093; -19.4557 -16.0629]
    alphas = [ -16.0629 -19.4557; -36.5093 -29.4557]
    policy = AlphaVectorPolicy(pomdp, alphas)

    # initial belief is 100% confidence in baby not being hungry
    @test isapprox(value(policy, b0), -16.0629)
    @test isapprox(value(policy, [1.0,0.0]), -16.0629)
    @test isapprox(actionvalues(policy, b0), [-16.0629, -19.4557])
    
    # because baby isn't hungry, policy should not feed (return false)
    @test action(policy, b0) == false
     
    # try pushing new vector
    push!(policy, [0.0,0.0], true)

    @test value(policy, b0) == 0.0
    @test action(policy, b0) == true
end
