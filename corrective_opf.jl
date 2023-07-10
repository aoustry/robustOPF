using Revise
using Random
using NLPModels
using Argos
using MadNLP
using MadNLPHSL
using Ipopt
using KNITRO
#using MathOptInterface
const MOI =MathOptInterface

const PS = ExaPF.PowerSystem

function generate_loads(pload_ref,qload_ref, nscen, magnitude)
    nbus = length(pload_ref);
    has_load = (pload_ref .> 0);
    Random.seed!(1);
    pload = magnitude .* (randn(nbus, nscen-1) .* has_load) .+ pload_ref
    qload = magnitude .* (randn(nbus, nscen-1) .* has_load) .+ qload_ref
    return hcat(pload_ref,pload), hcat(qload_ref,qload)
end

function instantiate!(blk::Argos.OPFModel)
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    return
end

function solve_corrective_opf(case,pl,ql,line_co,warm_start_dict,barrier_min=1e-10)
    #model = ExaPF.PolarForm(case)
    ev = Argos.CorrectiveEvaluator(case, pl, ql; line_constraints=line_co, epsilon=SMOOTHING_PARAM)
    blk = Argos.OPFModel(ev)    
    instantiate!(blk)
    if warm_start_dict["bool"]
        lenx0 = length(blk.meta.x0);
        blk.meta.x0[1:lenx0] .= warm_start_dict["x0"];
    end
    solver = MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        #linear_solver=Ma27Solver,
        max_iter=1000,
        print_level=MadNLP.DEBUG,
        tol=1e-9,
        mu_min = barrier_min,
        mu_init = barrier_min
    )
    MadNLP.solve!(solver)
    return (
        solver=solver,
        model=ev,
    )
end

function solve_corrective_opf_ipopt(case,pl,ql,line_co,warm_start_dict,barrier_min,scaling)
    #model = ExaPF.PolarForm(case)
    ev = Argos.CorrectiveEvaluator(case, pl, ql; line_constraints=line_co, epsilon=SMOOTHING_PARAM)
    blk = Argos.OPFModel(ev)    
    instantiate!(blk)
    knitro =false
    if knitro
        optimizer = KNITRO.Optimizer()  
        MOI.set(optimizer,MOI.RawOptimizerAttribute("bar_initmu"),barrier_min)
        MOI.set(optimizer,MOI.RawOptimizerAttribute("maxit"), 1000)

        if warm_start_dict["bool"]
            MOI.set(optimizer,MOI.RawOptimizerAttribute("maxit"), 50)
            MOI.set(optimizer,MOI.RawOptimizerAttribute("algorithm"),4)
            MOI.set(optimizer,MOI.RawOptimizerAttribute("strat_warm_start"),1)
        end
        #MOI.set(optimizer,MOI.RawOptimizerAttribute("scale"),3)
        #MOI.set(optimizer,MOI.RawOptimizerAttribute("objscalefactor"),scaling)

    else
        optimizer = Ipopt.Optimizer() 
        MOI.set(optimizer,MOI.RawOptimizerAttribute("max_iter"), 250)
        MOI.set(optimizer,MOI.RawOptimizerAttribute("mu_init"), 100*barrier_min)
        MOI.set(optimizer,MOI.RawOptimizerAttribute("mu_target"), barrier_min)
        MOI.set(optimizer,MOI.RawOptimizerAttribute("obj_scaling_factor"), scaling)
    end
    if warm_start_dict["bool"]
        output = Argos.optimize!(optimizer,ev,warm_start_dict["x0"]);
    else
        output = Argos.optimize!(optimizer,ev);
    end
    println(output.status)
    return (
        solver=0,
        obj_val = output.minimum,
        status = output.status,
        model=ev,
    )
end


function solve_corrective_tracking_opf(case, pl,ql, stack_ref,line_co)
    model = ExaPF.PolarForm(case)
    ev = Argos.CorrectiveEvaluator(model, pl, ql; line_constraints=line_co, tracking=true, stack_ref=stack_ref, epsilon=1e-5)
    blk = Argos.OPFModel(ev)
    instantiate!(blk)
    solver = MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        #linear_solver=Ma27Solver,
        max_iter=1000,
        print_level=MadNLP.ERROR,
        tol=1e-10,
    )
    MadNLP.solve!(solver)
    return (
        solver=solver,
        model=ev,
    )
end



#= nscen = 4
magnitude = 0.03
case = "../matpower/data/case9.m"

# For this example, we took as a reference MATPOWER's base solution
model = ExaPF.PolarForm(case)
stack = ExaPF.NetworkStack(model)
stack_ref = ExaPF.NetworkStack(model)

pload_ref = stack.pload |> Array
qload_ref = stack.qload |> Array
pl,ql = generate_loads(pload_ref,qload_ref, nscen, magnitude)
sol = solve_corrective_tracking_opf(case, pl,ql, stack_ref)


# Fetch solution
stack_sol = sol.model.stack

#[vmag ; vang ; pgen ; delta ; pgen_setpoint]
#nbus = length(qload_ref)
#ngen = model.network.ngen
#println("Objective value in Argos solver = ", sol.solver.obj_val)
#obj_val_test = (1/nscen)*0.5* (norm(stack_ref.vang .- stack_sol.vang[1:nbus],2)^2 + norm(stack_ref.vmag .- stack_sol.vmag[1:nbus])^2 + norm(stack_ref.pgen .- stack_sol.vuser[nscen+1:nscen+ngen])^2)
#println("Objective value computed from solution = ", obj_val_test)
 =#