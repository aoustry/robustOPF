using LinearAlgebra
using SuiteSparse



function softmax(x::Vector{Float64},ϵ::Float64)
    arg = ϵ^(-1) *x;
    M = maximum(arg);
    arg2 = arg .- M;
    return ϵ * (M .+ log(sum(exp.(arg2))));
end
function softmaxgrad(x::Vector{Float64},ϵ::Float64)
    arg = ϵ^(-1) *x;
    M = maximum(arg);
    arg2 = arg .- M;
    s = sum(exp.(arg2));
    return (exp.(arg2))/s;
end

function Evaluation(input_bar,pl,ql,aux_loadflow)
    """Input bar est le vecteur d'initalisation de la méthode de Newton """
    model_loadflow,stack_loadflow,constraints,jacx_loadflow,jacu_loadflow,jacx_ineq_constraints,jacu_ineq_constraints = aux_loadflow
    bmin, bmax = ExaPF.bounds(model_loadflow, stack_loadflow);
    gmin,gmax = ExaPF.bounds(model_loadflow, constraints);
    map_x = ExaPF.mapping(model_loadflow, State())
    nx = length(map_x);
    nu = length(jacu_ineq_constraints.map);
    map_x_no_delta = map_x[1:nx-1] 
    solver = ExaPF.NewtonRaphson(; verbose=0)
    nbus = length(stack_loadflow.pload);
    N = length(stack_loadflow.input)
    _, sizebatch = size(pl);
    val_res, grad_res= 0, zeros(nu);
    array = []
    for i in 1:sizebatch
        ### Setting of the stack in the set point and right scenario
        stack_loadflow.input[1:N] = input_bar;
        stack_loadflow.pload[1:nbus] = pl[:,i] ;
        stack_loadflow.qload[1:nbus] = ql[:,i] ;
        ### Load Flow
        ExaPF.set_params!(jacx_loadflow, stack_loadflow);
        ExaPF.nlsolve!(solver, jacx_loadflow, stack_loadflow);
        ExaPF.set_params!(jacu_loadflow, stack_loadflow);
        ExaPF.set_params!(jacx_ineq_constraints, stack_loadflow);
        ExaPF.set_params!(jacu_ineq_constraints, stack_loadflow);
        #Eval constraints and jacobian
        ineq_cons_eval = constraints(stack_loadflow);
        vector = [gmin - ineq_cons_eval;ineq_cons_eval - gmax;(bmin - stack_loadflow.input)[map_x_no_delta];(stack_loadflow.input - bmax)[map_x_no_delta]]        
        error = softmax(vector,ϵ_inner_scen);
        weights = softmaxgrad(vector,ϵ_inner_scen);
        #println(weights);
        ∇xg,∇ug =  ExaPF.jacobian!(jacx_loadflow, stack_loadflow),ExaPF.jacobian!(jacu_loadflow, stack_loadflow);
        ∇xh,∇uh =  ExaPF.jacobian!(jacx_ineq_constraints, stack_loadflow),ExaPF.jacobian!(jacu_ineq_constraints, stack_loadflow);
        aux_mat = hcat(I,zeros(nx-1));
        ∇xCons = vcat(-∇xh,∇xh,-aux_mat,aux_mat);
        ∇uCons = vcat(-∇uh,∇uh,zeros(2*(nx-1),nu));
        ∇xAggreg = weights'*∇xCons; ∇uAggreg = weights'*∇uCons;
        #Reduced space evaluation
        adjoint = (∇xg')\∇xAggreg';
        grad_reduced_space = ∇uAggreg[1,:] .- ∇ug'*adjoint;
        val_res+= -log(-error+tol_ineq);
        grad_res = grad_res .+ (1/abs(tol_ineq-error)) .* grad_reduced_space;
        array = [array ; error]
    end
    print(array);
    return val_res/sizebatch, grad_res/sizebatch
end

function EvaluationNoSmoothing(input_bar,pl,ql,aux_loadflow)
    """Input bar est le vecteur d'initalisation de la méthode de Newton """
    model_loadflow,stack_loadflow,constraints,jacx_loadflow,jacu_loadflow,jacx_ineq_constraints,jacu_ineq_constraints = aux_loadflow
    bmin, bmax = ExaPF.bounds(model_loadflow, stack_loadflow);
    gmin,gmax = ExaPF.bounds(model_loadflow, constraints);
    map_x = ExaPF.mapping(model_loadflow, State())
    nx = length(map_x);
    nu = length(jacu_ineq_constraints.map);
    map_x_no_delta = map_x[1:nx-1] 
    solver = ExaPF.NewtonRaphson(; verbose=0)
    nbus = length(stack_loadflow.pload);
    N = length(stack_loadflow.input)
    _, sizebatch = size(pl);
    val_res, grad_res= 0, zeros(nu);
    array = []
    for i in 1:sizebatch
        ### Setting of the stack in the set point and right scenario
        stack_loadflow.input[1:N] = input_bar;
        stack_loadflow.pload[1:nbus] = pl[:,i] ;
        stack_loadflow.qload[1:nbus] = ql[:,i] ;
        ### Load Flow
        ExaPF.set_params!(jacx_loadflow, stack_loadflow);
        ExaPF.nlsolve!(solver, jacx_loadflow, stack_loadflow);
        ExaPF.set_params!(jacu_loadflow, stack_loadflow);
        ExaPF.set_params!(jacx_ineq_constraints, stack_loadflow);
        ExaPF.set_params!(jacu_ineq_constraints, stack_loadflow);
        #Eval constraints and jacobian
        ineq_cons_eval = constraints(stack_loadflow);
        vector = [gmin - ineq_cons_eval;ineq_cons_eval - gmax;(bmin - stack_loadflow.input)[map_x_no_delta];(stack_loadflow.input - bmax)[map_x_no_delta]]        
        error = maximum(vector);
        #weights = softmaxgrad(vector,ϵ_inner_scen);
        #println(weights);
        #∇xg,∇ug =  ExaPF.jacobian!(jacx_loadflow, stack_loadflow),ExaPF.jacobian!(jacu_loadflow, stack_loadflow);
       # ∇xh,∇uh =  ExaPF.jacobian!(jacx_ineq_constraints, stack_loadflow),ExaPF.jacobian!(jacu_ineq_constraints, stack_loadflow);
        #aux_mat = hcat(I,zeros(nx-1));
        #∇xCons = vcat(-∇xh,∇xh,-aux_mat,aux_mat);
       # ∇uCons = vcat(-∇uh,∇uh,zeros(2*(nx-1),nu));
       # ∇xAggreg = weights'*∇xCons; ∇uAggreg = weights'*∇uCons;
        #Reduced space evaluation
       # adjoint = (∇xg')\∇xAggreg';
       # grad_reduced_space = ∇uAggreg[1,:] .- ∇ug'*adjoint;
        val_res+= -log(-error+tol_ineq);
       # grad_res = grad_res .+ (1/abs(tol_ineq-error)) .* grad_reduced_space;
       array = [array ; error]
    end
    print(array);
    return val_res/sizebatch, grad_res/sizebatch
end


function EvalXbatch(input_bar,sizebatch,pl,ql,aux_loadflow)
    """Input bar est le vecteur d'initalisation de la méthode de Newton """
    model_loadflow,stack_loadflow,_,jacx_loadflow,_,_,_ = aux_loadflow
    map_x = ExaPF.mapping(model_loadflow, State())
    solver = ExaPF.NewtonRaphson(; verbose=0)
    nbus = length(stack_loadflow.pload);
    N = length(stack_loadflow.input)
    array = []
    for i in 1:sizebatch
        ### Setting of the stack in the set point and right scenario
        stack_loadflow.input[1:N] = input_bar;
        stack_loadflow.pload[1:nbus] = pl[:,i] ;
        stack_loadflow.qload[1:nbus] = ql[:,i] ;
        ### Load Flow
        ExaPF.set_params!(jacx_loadflow, stack_loadflow);
        ExaPF.nlsolve!(solver, jacx_loadflow, stack_loadflow)
        array = vcat(array,stack_loadflow.input[map_x])
    end
    return array
end

#= case= "case9.m"
model = ExaPF.PolarFormRecourse(case,1)
basis = ExaPF.PolarBasis(model)
ngen = PS.get(model, PS.NumberOfGenerators())
nbus = PS.get(model, PS.NumberOfBuses())
stack_bar = ExaPF.NetworkStack(model)
pload_ref = stack_bar.pload |> Array
qload_ref = stack_bar.qload |> Array

aux_loadflow = build_aux_loadflow(model,basis,true)
law = LoadProbabilityLaw(pload_ref,qload_ref,0.3)
pbatch,qbatch = Oracle(stack_bar.input,law,100,aux_loadflow) 

Evaluation(stack_bar.input,pbatch,qbatch,aux_loadflow)   =#