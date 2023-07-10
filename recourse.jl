using Plots
using ExaPF
import ExaPF: AutoDiff
using SparseArrays
using LinearAlgebra
using Arpack
using Random
ϵ_response_function = 1e-3

const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers


function softmin(x::Vector{Float64},ϵ::Float64)
    arg = -ϵ^(-1) *x;
    M = maximum(arg);
    arg2 = arg .- M;
    return -ϵ * (M .+ log(sum(exp.(arg2))));
end
function softmingrad(x::Vector{Float64},ϵ::Float64)
    arg = -ϵ^(-1) *x;
    M = maximum(arg);
    arg2 = arg .- M;
    s = sum(exp.(arg2));
    return (exp.(arg2))/s;
end

function smoothResponseFunction(t::Float64,pmin::Float64, pmax::Float64,ϵ::Float64)
    @assert(pmax - ϵ >= pmin);
    if t >= pmax + 100*ϵ
        return pmax, 0
    end
    if t >= (pmax+pmin)/2
        return softmin([t,pmax],ϵ),softmingrad([t,pmax],ϵ)[1]
    end
    if t>= (pmin-100*ϵ)
        return -softmin([-t,-pmin],ϵ),softmingrad([-t,-pmin],ϵ)[1]
    end
    return pmin,0
end


#= function testingsmoothResponseFunction()
    pmin = -10.0
    pmax = -2.0
    interval = pmin-2:0.01:pmax+2
    answer = [smoothResponseFunction(t,pmin,pmax,1e-1) for t in interval]
    plot(interval,[tup[1] for tup in answer])
    plot!(interval,[tup[2] for tup in answer])
    title!("test");
    png("test.png");
end =#

function jacobian!(
    jac::AutoDiff.AbstractJacobian, stack,
)
    # init
    AutoDiff.set_value!(jac, stack.input)
    jac.t1sF .= 0.0
    # forward pass
    jac.func(jac.t1sF, jac.stack)
    # extract partials
    AutoDiff.partials!(jac)
    return jac.J
end

function Λ_u(Δ,ubar,u_α,u_min,u_max,nref,npv) 
    """Function encoding the value of the control vector u for a recourse Δ """

    "Voltage magnitude remains unchanged"
    @assert(length(ubar)==nref + 2 *npv);
    val_vector, grad_vector = 0.0 * ones(nref + 2 *npv),0.0 * ones(nref + 2 *npv);
    for k in 1:nref+npv
        val_vector[k] = ubar[k];
        grad_vector[k] = 0.0;
    end

    "Active power injections of the generators follow a (shifted and scaled) sigmoid"
    for k in nref+npv+1:nref+2*npv
        t = ubar[k] + u_α[k] * Δ;
        val, grad = smoothResponseFunction(t,u_min[k],u_max[k],ϵ_response_function);
        val_vector[k] = val;
        grad_vector[k] = u_α[k]*grad;
    end
    return (val_vector, grad_vector);
end

function Λ_pgref(Δ,pgref_bar,pgref_α,pmin_ref,pmax_ref)
    """Function encoding the value of the active power of the ref node for a recourse Δ """
    t = pgref_bar + pgref_α * Δ
    val, grad = smoothResponseFunction(t,pmin_ref,pmax_ref,ϵ_response_function);
    return val, pgref_α * grad
end

function extendedcontrolAndJacobian(stack,sizes,jacobians,extended_control_bounds,extended_control_bar,α,Δ)
    "Function computing the derivative of (x,Δ) ↦ G(x,Λ(̄u,Δ),p) with respect to x and with respect to Δ"
    " We emphasize that G is the concatenation of g and the active power flow balance at the ref node "
    #StartExtracting objects
    jx,ju,jx_power_gen_eval,ju_power_gen_eval=jacobians
    nx,nu,nref,npv = sizes
    u_min, u_max, pgref_min, pgref_max = extended_control_bounds;
    u_bar,pref_bar = extended_control_bar
    u_α, pgref_α = α
    #EndExtractingObjects
    valΛu, ∇Λu  = Λ_u(Δ,u_bar,u_α,u_min,u_max,nref,npv);
    stack.input[ju.map] .= valΛu;
    valΛpgref, ∇Λpgref = (Λ_pgref(Δ,pref_bar,pgref_α,pgref_min,pgref_max));
    aux  =     jx_power_gen_eval[1,:]
    jac_total_x = vcat(jacobian!(jx,stack),sparse(ones(length(aux.nzind)),aux.nzind,aux.nzval,1,nx));
    aux2  =     ju_power_gen_eval[1,:]
    vector_pref_u = sparse(ones(length(aux2.nzind)),aux2.nzind,aux2.nzval,1,nu);
    @assert(((vector_pref_u)*∇Λu)[1]==0.0);
    jacU = jacobian!(ju,stack);
    jac_total_Δ = vcat(jacU*∇Λu, vector_pref_u*∇Λu .- ∇Λpgref);
    return valΛu, valΛpgref,jac_total_x,jac_total_Δ
end

function extended_load_flow(stack,pflow,power_gen,sizes,jacobians,extended_control_bounds,extended_control_bar,α,perturbation)
    "Function solving the extended load flow G(x,Λ(̄u,Δ),p) = 0, for a given ̄u and a given p = perturbation"
    jx,_,_,_=jacobians
    Δ = 0.;
    x = view(stack.input, jx.map);
    nx = length(x);
    for i in 1:20
        valΛu, valΛpgref,jac_total_x,jac_total_Δ = extendedcontrolAndJacobian(stack,sizes,jacobians,extended_control_bounds,extended_control_bar,α,Δ);
        M = hcat(jac_total_x,jac_total_Δ)
        residual_ref = vcat(pflow(stack)+perturbation,power_gen(stack)[1] - valΛpgref);
        feaserror = norm(residual_ref);
        if feaserror<1e-10
            println("Success x(u); Norm residual = ", feaserror);
            println("Δ = ", Δ)
            println("Number of iterations = ",i)
            return 
        end
        z = M\(-residual_ref);
        x .= x .+ z[1:nx];
        Δ = Δ + z[nx+1];
    end
    println("No extended load flow solution corresponding to this perturbation")
end

function extended_load_flow_homotopy(stack,pflow,power_gen,sizes,jacobians,state_bounds,extended_control_bounds,extended_control_bar,α,perturbation_final,P)
    "Function solving the extended load flow G(x,Λ(̄u,Δ),p(t)) = 0, for a given ̄u and a given p = t*perturbation for t ∈ [0,1]
    The number of discretization for t is defined by P: t_k = k/P.
    We print the minimum eigenvalue of the extended Jacobian.
    "

    jx,_,_,_=jacobians
    x_min, x_max = state_bounds;
    Δ = 0.;
    x = view(stack.input, jx.map);
    nx = length(x);
    homotopy_counter = 1;
    perturbation = perturbation_final*(homotopy_counter/P);
    for i in 1:10*P
        valΛu, valΛpgref,jac_total_x,jac_total_Δ = extendedcontrolAndJacobian(stack,sizes,jacobians,extended_control_bounds,extended_control_bar,α,Δ);
        M = hcat(jac_total_x,jac_total_Δ)
        residual_ref = vcat(pflow(stack)+perturbation,power_gen(stack)[1] - valΛpgref);
        feaserror = norm(residual_ref);
        #println("Norm residual = ", feaserror);
        #println("Δ = ", Δ)
        if feaserror<1e-10
            if homotopy_counter==P
                println("Success x(u); Norm residual = ", feaserror);
                println("Number of iterations = ",i)
                return -1.0;
                return 
            else
                (eigvalues,_) = eigs(M'*M, nev=5,sigma = 0);
                println("State lower bounds = ", minimum(x-x_min),". State upper bounds = ", minimum(x_max-x));
                println("Homotopy Progress = ", floor(homotopy_counter/P *10000)/100, "% , Min eig Jacobian = ",minimum(eigvalues));
                homotopy_counter+=1;
                perturbation = perturbation_final*(homotopy_counter/P);
            end
        end
        try 
            z = M\(-residual_ref);
            x .= x .+ z[1:nx];
        Δ = Δ + z[nx+1];
        catch
            return min(minimum(x-x_min),minimum(x_max-x))
        end
        
    end
    println("No extended load flow solution corresponding to this perturbation")
    return -1.0;
end





