

abstract type LoadProbabilityLaw end 
#Structure of random laws to refine
mutable struct GaussianIndependantLaw <: LoadProbabilityLaw
    pload_ref::Vector{Float64}
    qload_ref::Vector{Float64}
    sigma::Float64
end

mutable struct GaussianIndependantLowRank<: LoadProbabilityLaw
    A::Matrix{Float64}
    rank::Int64
    pload_ref::Vector{Float64}
    qload_ref::Vector{Float64}
    sigma::Float64
end

mutable struct GaussianEqualCorrelation<: LoadProbabilityLaw
    d::Distribution 
    C::Matrix{Float64}
    pload_ref::Vector{Float64}
    qload_ref::Vector{Float64}
    sigma::Float64
end

function sample(law::GaussianIndependantLaw)
    nbus = length(law.pload_ref);
    has_load = (law.pload_ref .> 0);
    pload = law.sigma .* (randn(nbus) .* has_load) .+ law.pload_ref
    qload = law.sigma .* (randn(nbus) .* has_load) .+ law.qload_ref
    return pload,qload
end

function sample(law::GaussianIndependantLowRank)
    normal = randn(law.rank)
    while norm(normal,2)*norm(normal,2) > 2*law.rank
        normal = randn(law.rank)
    end
    mod_coeff = law.sigma .* (law.A*normal)
    #println(mod_coeff);
    pload =  law.pload_ref .* (1 .+ mod_coeff);
    qload = law.qload_ref .* (1 .+ mod_coeff);
    return pload,qload
end

#=function compute_std_sample(law::GaussianIndependantLowRank,SAMPLES::Integer)
    sum = 0
    for i in 1:SAMPLES
        normal = randn(law.rank)
        while norm(normal,2)*norm(normal,2) > 2*law.rank
            normal = randn(law.rank)
        end
        mod_coeff = law.sigma .* (law.A*normal)
        sum+= mod_coeff'*mod_coeff
    end
    return sqrt(sum/(SAMPLES*size(law.A,1)))
end=#

function compute_std_sample(law::GaussianIndependantLowRank,SAMPLES::Integer)
    logs = zeros(length(law.qload_ref))
    for i in 1:SAMPLES
        normal = randn(law.rank)
        while norm(normal,2)*norm(normal,2) > 2*law.rank
            normal = randn(law.rank)
        end
        mod_coeff = law.sigma .* (law.A*normal)
        logs .= logs .+ (mod_coeff .* mod_coeff)
    end
    logs .= logs/SAMPLES
    std_dev = sqrt.(logs)
    return sum(std_dev)/length(law.qload_ref)
end

function sample(law::GaussianEqualCorrelation)
    normal = rand(law.d)
    #while normal'*(law.C*normal) > 1
    #    normal = rand(law.d)
    #end
    mod_coeff = law.sigma .* (normal)
    #println(mod_coeff);
    pload =  law.pload_ref .* (1 .+ mod_coeff);
    qload = law.qload_ref .* (1 .+ mod_coeff);
    return pload,qload
end


function compute_std_sample(law::GaussianEqualCorrelation,SAMPLES::Integer)
    sum = 0
    for i in 1:SAMPLES
        normal = rand(law.d)
     #while normal'*(law.C*normal) > 1
     #   normal = rand(law.d)
    #end
        mod_coeff = law.sigma .* (normal)
        sum+= mod_coeff'*mod_coeff
    end
    return sqrt(sum/(SAMPLES*length(law.qload_ref)))
end



