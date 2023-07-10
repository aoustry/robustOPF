include("SQPsteps.jl")
include("evaluation.jl")
include("probalaw.jl")
include("oracle.jl")
include("corrective_opf.jl")
using LinearAlgebra

function merit_function(obj, g, Q,νg,νQ)
    return obj + νg* (norm(g)) + νQ * abs(Q);
end


""" Problem Parameters """
case = "../matpower/data/case30.m"
line_constraints = true;
magnitude = 0.02;


"""Problem Data """
model = ExaPF.PolarFormRecourse(case,1);
basis = ExaPF.PolarBasis(model)
map_x = ExaPF.mapping(model, State()); map_x_no_delta = map_x[1:length(map_x)-1] ; map_u = ExaPF.mapping(model, Control())
mapmain = [map_x_no_delta;map_u]
nx,nu = length(map_x), length(map_u); N = nx -1 + nu;
ngen = PS.get(model, PS.NumberOfGenerators())
nbus = PS.get(model, PS.NumberOfBuses())
stack_refcase = ExaPF.NetworkStack(model)
bmin, bmax = ExaPF.bounds(model, stack_refcase);
pload_refcase = stack_refcase.pload |> Array
qload_refcase = stack_refcase.qload |> Array
aux_loadflow = build_aux_loadflow(model,basis,line_constraints)


M= rand(nbus,nbus);
Q, R = qr(M);
A = copy(Q[:,1:8]);
law = GaussianIndependantLowRank(A,pload_refcase,qload_refcase,magnitude)
#law = GaussianIndependantLaw(pload_refcase,qload_refcase,magnitude)


"""Algo parameters"""
ϵ_inner_scen = 1e-5;
tol_ineq = 1e-3;
Ntries = 1e5;
size_batch = 5;
scale_obj = 0.0001;
ξ = 0.5; #Contraction rate for the vertical step in B-O algorithm
Δ0 = 1e-6; #Initial trust region (TR) diameter
increase_rate_Δ, decrease_rate_Δ = 1.1, 0.9 ; #Evolution rate for the TR diameter
Δmin = 1e-10; #Minimal value of the TR diameter (stopping criterion)
Δmax = 1e-2#1e4; #Maximal value of the TR diamter
νg = 1.0;
νQ = 1.0; 
 barrier_parameter = 0.01; 
 η = 1e-3;
 scale_g = 10.0;

"""Data structures """
stack_main = ExaPF.NetworkStack(model)
∂stack_main = ExaPF.NetworkStack(model)
cost = ExaPF.QuadraticCost(model) ∘ basis;
hess_cost = ExaPF.FullHessian(model, cost, mapmain);
pf_recourse = ExaPF.PowerFlowRecourse(model) ∘ basis;
jac_pf = ExaPF.Jacobian(model, pf_recourse, mapmain);
hess_pf = ExaPF.FullHessian(model, pf_recourse, mapmain);


"""Other variables """
dual = zeros(nx);
merit_main = Inf;

""" Initialization with random scenarios, solve the batch""" 
p_total,q_total = pload_refcase, qload_refcase
for i in 1:size_batch
    pl,ql = sample(law);
    p_total,q_total =hcat(p_total,pl),  hcat(q_total,ql)
end
stack_init = (solve_corrective_opf(case,p_total,q_total,line_constraints)).model.stack
stack_main.input[1:nbus] .= stack_init.vmag[1:nbus]
stack_main.input[nbus+1:2*nbus] .= stack_init.vang[1:nbus]
stack_main.input[2*nbus+2+ngen:end] .=stack_init.pgen[1:ngen]
outer_iter_counter = total_iter_counter = 0
Δ = Δ0;
#= """Begin Testing zone"""
value_barrier, gradient_barrier = Evaluation(stack_main.input,p_total,q_total,aux_loadflow);
for i in 1:10
    h = 1e-6*(randn(length(stack_main.input)))
    value_barrier2, gradient_barrier2 = Evaluation(stack_main.input + h,p_total,q_total,aux_loadflow);
    println(abs((value_barrier2-value_barrier)-gradient_barrier'*h[map_u])/abs(value_barrier2-value_barrier))
end
"""End testing zone """ =#

println("!!! Inclure η*pred !!!")

while true
    outer_iter_counter +=1
    inner_iter_counter = 0;
    p_newbatch,q_newbatch = Oracle(stack_main.input,law,size_batch,aux_loadflow,Ntries);
    
    if length(p_newbatch) == 0
        break
    end
    
    """Initialization inner loop """
    g = pf_recourse(stack_main);
    stack_proj = (solve_corrective_tracking_opf(case, hcat(pload_refcase,p_newbatch),hcat(qload_refcase,q_newbatch), stack_main,line_constraints)).model.stack
    sol_proj = [stack_proj.vmag[1:nbus];stack_proj.vang[1:nbus];zeros(1+ngen);stack_proj.pgen[1:ngen]]
    ∇Q = stack_main.input[mapmain] -  sol_proj[mapmain];
    Q = 0.5*norm(∇Q)^2 ;
    value_barrier, gradient_barrier = Evaluation(stack_main.input,p_total,q_total,aux_loadflow)
    merit_main = merit_function(cost(stack_main)[1]*scale_obj+barrier_parameter*value_barrier, g, Q,νg,νQ);
    predicted_progress = Inf
    while true
        println(outer_iter_counter," ", inner_iter_counter, "Δ = ", Δ);
        println("Q = ", Q);
        println("norm(g) = ", norm(g));
        println("Obj value = ",cost(stack_main)[1]," Barrier = ",value_barrier)
        #println("recourse_main_scen  =",stack_main.input[map_x[nx]])
        inner_iter_counter+=1; total_iter_counter+=1;
        """Functions and derivatives computations"""
        empty!(∂stack_main);
        ExaPF.adjoint!(cost, ∂stack_main, stack_main, scale_obj);
        ∇f = ∂stack_main.input[mapmain]; 
        λ_g = dual[1:nx] ;
        W = ExaPF.hessian!(hess_cost, stack_main,scale_obj) + ExaPF.hessian!(hess_pf, stack_main,λ_g)  ;
        ∇fandΩ = ∇f .+ (barrier_parameter .* [zeros(nx-1);gradient_barrier])
        ∇g = ExaPF.jacobian!(jac_pf,stack_main); 
        if Q < 1e-5 && norm(g)<1e-5 # && abs(predicted_progress)<1e-2
            Δ  = min(Δmax,  (increase_rate_Δ^2)*Δ);
            println("""Ligne testée, elle favorise les nouveaux scénarios """)
            stack_main.input[mapmain] = sol_proj[mapmain]
            break
        end
        A = vcat(scale_g*∇g,∇Q');
        b = vcat(scale_g*g,Q);
        """Bounds limits """
        d♭,d♯ = (bmin.-stack_main.input)[mapmain] ,(bmax .-stack_main.input)[mapmain] 
        """Normal step"""
        vdir = normal_step(A,b,d♭,d♯,ξ,Δ,N);
        """Tangential step"""
        diff_primal,dual_new,predicted_progress = tangential_step(W,∇fandΩ,A,d♭,d♯,Δ,N,vdir);
        """Update variables """
        copy_main = copy(stack_main.input)
        stack_main.input[mapmain]  = stack_main.input[mapmain] + diff_primal; 
        """Evaluation new point"""
        gnew = pf_recourse(stack_main);
        stack_proj = (solve_corrective_tracking_opf(case, hcat(pload_refcase,p_newbatch),hcat(qload_refcase,q_newbatch), stack_main,line_constraints)).model.stack
        sol_proj = [stack_proj.vmag[1:nbus];stack_proj.vang[1:nbus];zeros(1+ngen);stack_proj.pgen[1:ngen]]
        ∇Qnew = stack_main.input[mapmain] -  sol_proj[mapmain];
        Qnew = 0.5*norm(∇Q)^2 ;
        success_condition = false; merit = undef; value_barrier_new = undef; gradient_barrier_new =undef;
        try
            value_barrier_new, gradient_barrier_new = Evaluation(stack_main.input,p_total,q_total,aux_loadflow);
            merit = merit_function(cost(stack_main)[1]*scale_obj+barrier_parameter*value_barrier_new, gnew, Qnew,νg,νQ);
            success_condition  = merit < merit_main  #+ η * pred
            println("Conv", cost(stack_main)[1], value_barrier_new, norm(gnew), Qnew)
        catch
            println("Fail")
            success_condition = false;
        end
        if success_condition
            println(predicted_progress," ",merit) ;
            merit_main = merit;
            Q, ∇Q = Qnew, ∇Qnew; 
            value_barrier, gradient_barrier = value_barrier_new, gradient_barrier_new;
            g = gnew;
            dual = dual_new;
            Δ  = min(Δmax,  increase_rate_Δ*Δ);
        else
            """ We cancel the update"""
            stack_main.input .= copy_main;
            #stack_main.input[mapmain] .= copy_main[mapmain];
            Δ  = max(Δmin,  decrease_rate_Δ*Δ);
            if Δ<=Δmin
                break
            end
        end
    end
    p_total,q_total = hcat(p_total, p_newbatch), hcat(q_total, q_newbatch);
    if Δ<=Δmin
        break
    end
end



#testing = (solve_corrective_opf(case,p_total,q_total,line_constraints))