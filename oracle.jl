
using Arpack
using SparseArrays
using KLU
const LS = ExaPF.LinearSolvers


function spectral_evaluation(JtopJ)
    nx = size(JtopJ, 1)
    eigvalues,_ = eigs(JtopJ,nev = 1, sigma = 0, v0=ones(nx));
    return sqrt(minimum(eigvalues))
end


function pvalue(N,p0,success)
    sum = 0
    for j in 0:success
        sum+= p0^j *(1-p0)^(N-j)*binomial(N,j)
    end
    println("pvalue for rejection p ≥ p0 =",sum)
end

struct OracleEvaluator
    batch_model::ExaPF.PolarFormRecourse
    jac::ExaPF.Jacobian
    stack::ExaPF.NetworkStack
    constraints::ExaPF.ComposedExpressions
    solver::ExaPF.NewtonRaphson
    linear_solver::Any
end

function OracleEvaluator(model_ext_oracle, k; line_constraints=false)
    batch_model = ExaPF.PolarFormRecourse(model_ext_oracle.network, CPU(), k)
    mapx = ExaPF.mapping(model_ext_oracle, State())
    stack_ext_oracle = ExaPF.NetworkStack(batch_model)
    basis = ExaPF.PolarBasis(batch_model)
    pf_recourse = ExaPF.PowerFlowRecourse(batch_model, epsilon=SMOOTHING_PARAM) ∘ basis
    jacx_ext_oracle = ExaPF.Jacobian(batch_model, pf_recourse, mapx)
    constraints_expr = Any[
        ExaPF.ReactivePowerBounds(batch_model),
    ]
    if line_constraints
        push!(constraints_expr, ExaPF.LineFlows(batch_model))
    end
    constraints = ExaPF.MultiExpressions(constraints_expr) ∘ basis
    solver = ExaPF.NewtonRaphson(; verbose=0)

    linear_solver = LS.DirectSolver(klu(jacx_ext_oracle.J))
    return OracleEvaluator(batch_model, jacx_ext_oracle, stack_ext_oracle, constraints, solver, linear_solver)
end

function LS.update!(s::LS.DirectSolver{KLU.KLUFactorization{Tv, Ti}}, J::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    klu!(s.factorization, J)
end

function get_feasibility_bounds!(inf_pr, vmag, vmag_min, vmag_max, pq, nbus, k)
    @inbounds for i in 1:k
        for j in pq
            dlb = vmag_min[j] - vmag[j + (i-1) * nbus]
            dub = vmag[j + (i-1) * nbus] - vmag_max[j]
            inf_pr[k] = max(inf_pr[k], dlb, dub)
        end
    end
    return
end

function get_feasibility_constraints!(inf_pr, g, gmin, gmax, k)
    m = div(length(gmin), k)
    @inbounds for i in 1:k
        for j in 1:m
            dlb = gmin[j] - g[j + (i-1) * m]
            dub = g[j + (i-1) * m] - gmax[j]
            inf_pr[k] = max(inf_pr[k], dlb, dub)
        end
    end
    return
end

function assess!(
    model::ExaPF.PolarFormRecourse,
    ev::OracleEvaluator,
    input_bar,
    law::LoadProbabilityLaw,
    batch_size::Integer,
    N_tries,
    threshold,
    threadid,
)

    vmag_min, vmag_max = PS.bounds(model.network, PS.Buses(), PS.VoltageMagnitude())
    gmin, gmax = ExaPF.bounds(ev.batch_model, ev.constraints)
    cons_eval = similar(gmin)

    inf_pr = zeros(model.k)
    nbus = length(ev.stack.pload)
    success_threshold, success, tries = (0, 0, 0)

    pl_batch, ql_batch, error_batch = zeros(nbus, 0),zeros(nbus, 0),Float64[]
    pl_log, ql_log, error_log = zeros(nbus, 0),zeros(nbus, 0),Float64[]
    N = length(ev.stack.input)
    while success_threshold < batch_size && (tries < N_tries)
        tries += ev.batch_model.k
        pl, ql = sample(law, ev.batch_model.k)
        ev.stack.input[1:N] .= input_bar

        ExaPF.set_params!(ev.stack, pl, ql)
        ExaPF.set_params!(ev.jac, ev.stack)
        ExaPF.nlsolve!(ev.solver, ev.jac, ev.stack; linear_solver=ev.linear_solver)

        ev.constraints(cons_eval, ev.stack)
        fill!(inf_pr, 0.0)
        get_feasibility_bounds!(inf_pr, ev.stack.vmag, vmag_min, vmag_max, model.network.pq, nbus, model.k)
        get_feasibility_constraints!(inf_pr, cons_eval, gmin, gmax, model.k)

        infeas = findall(inf_pr .> threshold)
        if length(infeas) >= 1
            success_threshold += length(infeas)
            pl_batch = hcat(pl_batch, pl[:, infeas])
            ql_batch = hcat(ql_batch, ql[:, infeas])
            push!(error_batch, inf_pr[infeas])
        end

        infeas_ineq = findall(threshold .> inf_pr .> tol_ineq)
        if length(infeas_ineq) >= 1
            success += length(infeas_ineq)
            push!(error_log, inf_pr[infeas_ineq]...)
            pl_log = hcat(pl_log, pl[:, infeas_ineq])
            ql_log = hcat(ql_log, ql[:, infeas_ineq])
        end
    end

    len = success_threshold
    if (success_threshold<batch_size) && length(error_log)>0
        #Complement with smaller
        tuples = [(e,i) for (i,e) in enumerate(error_log)];
        sorted = sort(tuples,rev=true);
        complement = min(length(error_log),batch_size-success_threshold)
        indexes = [sorted[aux][2] for aux in 1:complement]
        len+=complement;
        pl, ql = pl_log[:,indexes] , ql_log[:,indexes];
        error_array = error_log[indexes]
        if length(pl_batch)==0
            pl_batch = pl
            ql_batch = ql
            error_batch = error_array;
        else
            pl_batch = hcat(pl_batch, pl);
            ql_batch = hcat(ql_batch, ql);
            error_batch = vcat(error_batch,error_array)
        end
    end
    println("Oracle : tries = ",tries, ", new scenarios added = ", len,"." )
    res = Dict("pl_batch"=>pl_batch,"ql_batch"=>ql_batch,"error_batch"=>error_batch,"len"=>len,"boolean"=>(success_threshold>0),"success"=>success,"tries"=>tries,"eigvals"=>eigvals)
    return res
end

function OracleParallel(model, evaluators, input_bar,law_array,batch_size::Integer,N_tries,threshold)
    res = [Dict() for i in 1:Kthreads]
    small_batch_size = Integer(max(1,round(batch_size/Kthreads)))
    #for i = 1:Kthreads
    Threads.@threads for i = 1:Kthreads
        res[i] = assess!(model, evaluators[i], input_bar,law_array[i],small_batch_size,N_tries/Kthreads,threshold,Threads.threadid())
    end
    len = minimum([batch_size,sum(dico["len"] for dico in res)])
    success = sum(dico["success"] for dico in res)
    tries = sum(dico["tries"] for dico in res)
    measured_ratio = success/tries
    tuples = [(e,i,j) for i in 1:Kthreads for (j,e) in enumerate(res[i]["error_batch"])]
    sorted = sort(tuples,rev=true);
    boolean = maximum([dico["boolean"] for dico in res])
    eigvals = res[1]["eigvals"]
    output = Dict("ratio"=>measured_ratio,"len"=>len,"success"=>success,"tries"=>tries,"boolean"=>boolean,"error_batch"=>[res[s[2]]["error_batch"][s[3]] for s in sorted[1:len]],"eigvals"=>eigvals)
    if len >0
        output["pl_batch"] = reduce(hcat,[res[s[2]]["pl_batch"][:,s[3]] for s in sorted[1:len]]);
        output["ql_batch"] = reduce(hcat,[res[s[2]]["ql_batch"][:,s[3]] for s in sorted[1:len]]);
    else
        output["pl_batch"] = [];
        output["ql_batch"] = [];
    end
    return output
end
