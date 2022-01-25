using SpecialFunctions: digamma
const AbstractTens3{T} = AbstractArray{T,3};

struct TimeVaryingHMM{D <: Distribution,T}
    a::Vector{T}
    A::Array{T,3}
    B::Matrix{D}
    TimeVaryingHMM{D,T}(a, A, B) where {D,T} = assert_timevaryinghmm(a, A, B) && new(a, A, B)
end

TimeVaryingHMM(
    a::AbstractVector{T},
    A::AbstractArray{T,3},
    B::AbstractMatrix{D}
) where {D,T} = TimeVaryingHMM{D,T}(a, A, B)

TimeVaryingHMM(A::AbstractArray{T,3}, B::AbstractMatrix{D}) where {D,T} =
    TimeVaryingHMM{F,T}(ones(size(A, 1)) ./ size(A, 1), A, B)

function assert_timevaryinghmm(a::AbstractVector, A::AbstractTens3, B::AbstractMatrix{<:Distribution})
    @argcheck isprobvec(a)
    @argcheck istransmats(A)
    @argcheck all(length.(B) .== length(B[1])) ArgumentError("All distributions must have the same dimensions")
    @argcheck size(A,3) + 1 == size(B,2) ArgumentError("Number of transition rates must match length of chain")
    @argcheck length(a) == size(A,1) == size(B,1)
    return true
end

istransmats(A::AbstractTens3) = all(i->istransmat(@view A[:,:,i]), 1:size(A,3))
    
==(h1::TimeVaryingHMM, h2::TimeVaryingHMM) = (h1.a == h2.a) && (h1.A == h2.A) && (h1.B == h2.B)

function rand(
    rng::AbstractRNG,
    hmm::TimeVaryingHMM;
    init = rand(rng, Categorical(hmm.a)),
    seq = false,
)
    T = size(hmm.B, 2)
    z = Vector{Int}(undef, T)
    (T >= 1) && (z[1] = init)
    for t = 2:T
        z[t] = rand(rng, Categorical(hmm.A[z[t-1],:,t-1]))
    end
    y = randobs(rng, hmm, z)
    seq ? (z, y) : y
end

function randobs(rng::AbstractRNG, hmm::TimeVaryingHMM{<:UnivariateDistribution}, z::AbstractVector{<:Integer})
    y = Vector{Float64}(undef, length(z))
    for t in eachindex(z)
        y[t] = rand(rng, hmm.B[z[t],t])
    end
    y
end

size(hmm::TimeVaryingHMM, dim = :) = (size(hmm.B, 1), length(hmm.B[1]), size(hmm.B, 2))[dim]
copy(hmm::TimeVaryingHMM) = TimeVaryingHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

function nparams(hmm::TimeVaryingHMM)
    (length(hmm.a) - 1) + (size(hmm.A,1) * size(hmm.A,2) - size(hmm.A,1)) * size(hmm.A,3)
end


###

function loglikelihoods!(LL::AbstractTens3, hmm::TimeVaryingHMM{<:UnivariateDistribution}, 
                         observations::AbstractMatrix)
    T, N = size(observations)
    K = size(hmm, 1)
    @argcheck size(LL) == (T, K, N)
    @argcheck T == size(hmm, 3)
    for n in 1:N
        for i in 1:K, t in 1:T
            LL[t,i,n] = logpdf(hmm.B[i,t], observations[t,n])
        end
    end

    LL
end

function get_γ!(γ::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix)
    @argcheck size(α) == size(β) == size(γ)

    T, K = size(γ)

    for t in 1:T
        γ[t,:] .= α[t,:] .* β[t,:]
        Z = sum(@view γ[t,:])
        
        for k in 1:K
            γ[t,k] /= Z
        end
    end

    γ
end

function get_ξ!(ξ::AbstractTens3, α::AbstractMatrix, β::AbstractMatrix, 
                A::AbstractTens3, LL::AbstractMatrix)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1) + 1
    @argcheck size(α, 2) == size(LL, 2) == size(β, 2) == size(ξ, 2) == size(ξ, 3)

    T, K = size(LL)
    @argcheck T == size(A,3) + 1

    for t in 1:T-1
        m = vec_maximum(@view LL[t+1, :])
        
        sm = zero(eltype(ξ))
        for i in 1:K, j in 1:K
            ξ[t,i,j] = α[t,i] * A[i,j,t] * exp(LL[t+1,j] - m) * β[t+1,j]
            sm += ξ[t,i,j]
        end

        ξ[t,:,:] ./= sm
    end

    ξ
end

####

function update_A!(
    A::AbstractTens3,
    ξ::AbstractArray
)
    @argcheck size(ξ, 1) == size(A,3) 
    @argcheck size(A, 1) == size(A, 2) == size(ξ, 2) == size(ξ, 3)

    T = size(ξ, 1) + 1
    K = size(ξ, 2)
    N = size(ξ, 4)

    fill!(A, zero(eltype(A)))

    for t in 1:T-1
        for i in 1:K
            sm = zero(eltype(A))

            for n in 1:N, j in 1:K
                A[i,j,t] += ξ[t,i,j,n]
                sm += ξ[t,i,j,n]
            end

            if iszero(sm)
                @warn "No transitions happened here: t=$(t)"

                A[i,:,t] .= one(eltype(A)) / K
            else
                A[i,:,t] ./= sm
            end
        end
    end

    A
end

function estimator_NB(comp::NegativeBinomial, observations::AbstractVector{Int}, zz::AbstractVector;
                      p_eps=1e-3, r_eps=1e-3)
    N = length(observations)
    
    dig = digamma(comp.r + 1e-12)

    sum_zz_delta = 0.0
    sum_zz_samples = 0.0
    sum_zz = 0.0

    for j in 1:N
        sum_zz_delta += zz[j] * comp.r * (digamma(comp.r + observations[j] + 1e-12) - dig)
        sum_zz_samples += zz[j] * observations[j]
        sum_zz += zz[j]
    end

    lambda = sum_zz_delta / sum_zz 
    beta = 1 - 1 / (1 - comp.p + 1e-12) - 1 / (log(comp.p + 1e-12) + 1e-12)

    theta = beta * sum_zz_delta / (sum_zz_samples - (1 - beta) * sum_zz_delta)

    theta = clamp(theta, p_eps, 1 - p_eps)

    new_r = -lambda / log(theta)
    new_r = max(new_r, r_eps)

    new_p = theta

    NegativeBinomial(new_r, new_p)
end

# In-place update of the observations distributions.
function update_B!(B::AbstractMatrix, observations, γ::AbstractTens3)
    T = size(γ, 1)
    K = size(γ, 2)
    N = size(γ, 3)
    
    @argcheck T == size(observations, 1) == size(B, 2)
    @argcheck N == size(observations, 2)
    @argcheck K == size(B, 1)
    
    Threads.@threads for t in 1:T
        for i in 1:K
            if sum(γ[t,i,:]) <= 0
                continue
            end
            
            B[i,t] = estimator_NB(B[i,t], view(observations, t, :), view(γ, t, i, :))
        end
    end
    
    B
end

function fit_mle!(
    hmm::TimeVaryingHMM,
    observations::AbstractMatrix;
    display = :none,
    maxiter = 100,
    tol = 1e-3,
    robust = false,
    estimator = fit_mle,
    fit_dists = false
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    @argcheck fit_dists === false "Not yet supported"

    @assert !fit_dists

    T, N, K = size(observations, 1), size(observations, 2), size(hmm, 1)
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates
    c = zeros(T, N)
    α = zeros(T, K, N)
    β = zeros(T, K, N)
    γ = zeros(T, K, N)
    ξ = zeros(T-1, K, K, N)
    LL = zeros(T, K, N)

    loglikelihoods!(LL, hmm, observations)

    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    Threads.@threads for n in 1:N
        forwardlog!(view(α,:,:,n), view(c,:,n), hmm.a, hmm.A, view(LL,:,:,n))
        backwardlog!(view(β,:,:,n), view(c,:,n), hmm.a, hmm.A, view(LL,:,:,n))
        get_γ!(view(γ,:,:,n), view(α,:,:,n), view(β,:,:,n))
        get_ξ!(view(ξ,:,:,:,n), view(α,:,:,n), view(β,:,:,n), hmm.A, view(LL,:,:,n))
    end

    logtot = sum(c)
    push!(history.logtots, logtot)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_A!(hmm.A, ξ)
        update_B!(hmm.B, observations, γ)

        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)

        Threads.@threads for n in 1:N
            forwardlog!(view(α,:,:,n), view(c,:,n), hmm.a, hmm.A, view(LL,:,:,n))
            backwardlog!(view(β,:,:,n), view(c,:,n), hmm.a, hmm.A, view(LL,:,:,n))
            get_γ!(view(γ,:,:,n), view(α,:,:,n), view(β,:,:,n))
            get_ξ!(view(ξ,:,:,:,n), view(α,:,:,n), view(β,:,:,n), hmm.A, view(LL,:,:,n))
        end

        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        push!(history.logtots, logtotp)
        history.iterations += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history.converged = true
            break
        end

        logtot = logtotp
    end

    if !history.converged
        if display in [:iter, :final]
            @warn "EM has not converged after $(history.iterations) iterations, logtot = $logtot"
        end
    end

    history
end



# In-place forward pass, where α and c are allocated beforehand.
function forwardlog!(
    α::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractTens3,
    LL::AbstractMatrix,
)
    @argcheck size(α, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(α, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    T, K = size(LL)
    @argcheck T == size(A,3) + 1

    m = vec_maximum(view(LL, 1, :))

    fill!(α, zero(eltype(α)))

    sm = zero(eltype(α))
    for i in 1:K
        α[1,i] = a[i] * exp(LL[1,i] - m)
        sm += α[1,i]
    end
    
    α[1,:] ./= sm

    c[1] = log(sm) + m

    for t = 2:T
        m = vec_maximum(@view LL[t,:])

        sm = zero(eltype(α))
        for j in 1:K
            for i in 1:K
                α[t,j] += A[i,j,t-1] * α[t-1,i]
            end

            α[t,j] *= exp(LL[t,j] - m)
            sm += α[t,j]
        end

        α[t,:] ./= sm

        c[t] = log(sm) + m
    end

    α, c
end

# In-place backward pass, where β and c are allocated beforehand.
function backwardlog!(
    β::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractTens3,
    LL::AbstractMatrix,
)
    @argcheck size(β, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(β, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    T, K = size(LL)
    @argcheck T == size(A,3) + 1
    (T == 0) && return

    fill!(β, zero(eltype(β)))

    β[end,:] .= one(eltype(β))

    for t = T-1:-1:1
        m = vec_maximum(@view LL[t+1,:])

        sm = zero(eltype(β))
        for j in 1:K
            for i in 1:K
                β[t,j] += β[t+1,i] * A[j,i,t] * exp(LL[t+1,i] - m)
            end

            sm += β[t, j]
        end

        β[t, :] ./= sm

        c[t+1] = log(sm) + m
    end

    m = vec_maximum(@view LL[1, :])

    sm = zero(eltype(β))

    for j in 1:K
        sm += a[j] * exp(LL[1, j] - m) * β[1, j]
    end

    c[1] = log(sm) + m

    β, c
end

# In-place posterior computation, where γ is allocated beforehand.
function posteriors!(γ::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix, c::AbstractVector)
    @argcheck size(γ) == size(α) == size(β)
    T, K = size(α)
    for t in OneTo(T)
        sm = zero(eltype(γ))

        for i in 1:K
            γ[t, i] = α[t, i] * β[t, i] * c[t]
            sm += γ[t, i]/posteriors
        end

        for i in OneTo(K)
            γ[t, i] /= c
        end
    end
end


function loglikelihood(hmm::TimeVaryingHMM, observations; robust = false)
    T, N, K = size(observations, 1), size(observations, 2), size(hmm, 1)
    @argcheck T == size(hmm.B, 2)
    m = Matrix{Float64}(undef, T, K)
    c = Vector{Float64}(undef, T)
    LL = Array{Float64}(undef, T, K, 1)

    ret = 0.0
    for n in 1:N
        loglikelihoods!(LL, hmm, @view observations[:,n:n])
        forwardlog!(m, c, hmm.a, hmm.A, @view LL[:,:,1])
        ret += sum(c)
    end

    ret
end