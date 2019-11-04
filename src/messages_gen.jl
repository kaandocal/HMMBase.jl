# Convenience functions

# The following methods are generated by the first macro:
# forward(a, A, L)               -> α, logtot
# forward(hmm, observation)      -> α, logtot
#
# backward(a, A, L)              -> β, logtot
# backward(hmm, observations)    -> β, logtot

# The following methods are also defined:
# posteriors(α, β)                 -> γ
# posteriors(a, A, L)              -> γ
# posteriors(hmm, observations)    -> γ

# Forward/Backward

for f in (:forward, :backward)
    f!  = Symbol("$(f)!")   # forward!
    fl! = Symbol("$(f)log!")  # forwardlog!
    @eval begin
        # [forward,backward](a, A, L)
        """
            $($f)(a, A, L)

        Compute $($f) probabilities using samples likelihoods.
        See [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
        """
        function $(f)(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; logl = false)
            m = Matrix{Float64}(undef, size(L))
            c = Vector{Float64}(undef, size(L)[1])
            if logl
                $(fl!)(m, c, a, A, L)
            else
                warn_logl(L)
                $(f!)(m, c, a, A, L)
            end
            m, sum(log.(c))
        end

        # [forward,backward](hmm, observations)
        """
            $($f)(hmm, observations)

        # Example
        ```julia
        hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
        z, y = rand(hmm, 1000)
        probs, tot = $($f)(hmm, y)
        ```
        """
        function $(f)(hmm::AbstractHMM, observations; logl = false)
            L = likelihoods(hmm, observations, logl = logl)
            $(f)(hmm.a, hmm.A, L, logl = logl)
        end
    end
end

# Posteriors

"""
    posteriors(α, β)

Compute posterior probabilities from `α` and `β`.
"""
function posteriors(α::AbstractMatrix, β::AbstractMatrix)
    γ = Matrix{Float64}(undef, size(α))
    posteriors!(γ, α, β)
    γ
end

"""
    posteriors(a, A, L)

Compute posterior probabilities using samples likelihoods.
"""
function posteriors(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; kwargs...)
    α, _ = forward(a, A, L; kwargs...)
    β, _ = backward(a, A, L; kwargs...)
    posteriors(α, β)
end

"""
    posteriors(hmm, observations)

Compute posterior probabilities using samples likelihoods.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
γ = posteriors(hmm, y)
```
"""
function posteriors(hmm::AbstractHMM, observations; logl = false)
    L = likelihoods(hmm, observations, logl = logl)
    posteriors(hmm.a, hmm.A, L, logl = logl)
end

function warn_logl(L::AbstractMatrix)
    if any(L .< 0)
        @warn "Negative likelihoods values, use the `logl = true` option if you are using log-likelihoods."
    end
end
