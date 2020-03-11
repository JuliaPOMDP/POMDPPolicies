

# exploration schedule 
"""
    ExplorationSchedule
Abstract type for exploration schedule. 
It is useful to define the schedule of a parameter of an exploration policy.
The effect of a schedule is defined by the `update_value` function.
"""
abstract type ExplorationSchedule <: Function end 

"""
    update_value(::ExplorationSchedule, value)
Returns an updated value according to the schedule.
"""
function update_value(::ExplorationSchedule, value) end


"""
    LinearDecaySchedule
A schedule that linearly decreases a value from `start_val` to `end_val` in `steps` steps.
if the value is greater or equal to `end_val`, it stays constant.

# Constructor 

`LinearDecaySchedule(;start, stop, steps)`
"""
@with_kw struct LinearDecaySchedule{R<:Real} <: ExplorationSchedule
    start::R
    stop::R
    steps::Int
end

function (schedule::LinearDecaySchedule)(k)
    rate = (schedule.start - schedule.stop) / schedule.steps
    val = schedule.start - k*rate 
    val = max(schedule.stop, val)
end


"""
    ExplorationPolicy <: Policy
An abstract type for exploration policies.
Sampling from an exploration policy is done using `action(exploration_policy, on_policy, state)`
"""
abstract type ExplorationPolicy <: Policy end

# """
#     exploration_parameter(::ExplorationPolicy)
# returns the exploration parameter of an exploration policy, e.g. epsilon for e-greedy or temperature for softmax
# """
# function exploration_parameter end

"""
    EpsGreedyPolicy <: ExplorationPolicy

represents an epsilon greedy policy, sampling a random action with a probability `eps` or returning an action from a given policy otherwise.
The evolution of epsilon can be controlled using a schedule. This feature is useful for using those policies in reinforcement learning algorithms. 

constructor:

`EpsGreedyPolicy(problem::Union{MDP, POMDP}, eps::Float64; rng=Random.GLOBAL_RNG, schedule=ConstantSchedule)`
"""
struct EpsGreedyPolicy{T<:Function, R<:AbstractRNG, A} <: ExplorationPolicy
    eps::T
    rng::R
    actions::A
end

function EpsGreedyPolicy(problem::Union{MDP, POMDP}, eps::Function; 
                         rng::AbstractRNG=Random.GLOBAL_RNG)
    return EpsGreedyPolicy(eps, rng, actions(problem))
end
function EpsGreedyPolicy(problem::Union{MDP, POMDP}, eps::Real; 
                         rng::AbstractRNG=Random.GLOBAL_RNG)
    return EpsGreedyPolicy(x->eps, rng, actions(problem))
end


function POMDPs.action(p::EpsGreedyPolicy, on_policy::Policy, k, s)
    if rand(p.rng) < p.eps(k)
        return rand(p.rng, p.actions)
    else 
        return action(on_policy, s)
    end
end

# exploration_parameter(p::EpsGreedyPolicy, k) = p.eps(k)

# softmax 
"""
    SoftmaxPolicy <: ExplorationPolicy

represents a softmax policy, sampling a random action according to a softmax function. 
The softmax function converts the action values of the on policy into probabilities that are used for sampling. 
A temperature parameter can be used to make the resulting distribution more or less wide.
"""
struct SoftmaxPolicy{T<:Function, R<:AbstractRNG, A} <: ExplorationPolicy
    temperature::T
    rng::R
    actions::A
end

function SoftmaxPolicy(problem, temperature::Function; 
                       rng::AbstractRNG=Random.GLOBAL_RNG)
    return SoftmaxPolicy(temperature, rng, actions(problem))
end
function SoftmaxPolicy(problem, temperature::Real; 
                       rng::AbstractRNG=Random.GLOBAL_RNG)
    return SoftmaxPolicy(x->temperature, rng, actions(problem))
end

function POMDPs.action(p::SoftmaxPolicy, on_policy::Policy, k, s)
    vals = actionvalues(on_policy, s)
    vals ./= p.temperature(k)
    maxval = maximum(vals)
    exp_vals = exp.(vals .- maxval)
    exp_vals /= sum(exp_vals)
    return p.actions[sample(p.rng, Weights(exp_vals))]
end

# exploration_parameter(p::SoftmaxPolicy, k) = p.temperature(k)
