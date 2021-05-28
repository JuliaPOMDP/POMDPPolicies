"""
    BlindPolicySolver <: Solver
POMDP solver for solving a blind policy, which selects the same action regardless of the current belief state.
"""
mutable struct BlindPolicySolver <: Solver
    max_iterations::Int64
    tolerance::Float64
    verbose::Bool
end
function BlindPolicySolver(;max_iterations::Int64=100, tolerance::Float64=1e-3, verbose::Bool=false)
    return BlindPolicySolver(max_iterations, tolerance, verbose)
end

function POMDPs.solve(solver::BlindPolicySolver, pomdp::POMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)

    ns = length(S)
    na = length(A)

    γ = discount(pomdp)

    # best action worst state lower bound
    α_init = 1 / (1 - γ) * maximum(minimum(reward(pomdp, s, a, sp) for s in S for sp in S) for a in A)
    Γ = fill(α_init, ns, na)

    for i = 1:solver.max_iterations
        residual = 0.0
        for (ai, a) in enumerate(A)
            for (si, s) in enumerate(S)

                sp_dist = transition(pomdp, s, a)

                r = 0.0
                for (sp, p_sp) in weighted_iterator(sp_dist)
                    r += p_sp * (reward(pomdp, s, a, sp) + γ * Γ[stateindex(pomdp, sp), ai])
                end

                alpha_diff = abs(Γ[si, ai] - r)
                Γ[si, ai] = r
                residual = max(alpha_diff, residual)
            end
        end
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G \n", i, residual) : nothing
        residual < solver.tolerance ? break : nothing
    end

    return AlphaVectorPolicy(pomdp, Γ, A)
end