
using Arpack


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



function build_aux_loadflow(model_ext_oracle,basis,line_constraints)
    mapx = ExaPF.mapping(model_ext_oracle, State())
    mapu = ExaPF.mapping(model_ext_oracle, Control())
    stack_ext_oracle = ExaPF.NetworkStack(model_ext_oracle)
    pf_recourse = ExaPF.PowerFlowRecourse(model_ext_oracle,epsilon = SMOOTHING_PARAM) ∘ basis
    jacx_ext_oracle = ExaPF.Jacobian(model_ext_oracle, pf_recourse, mapx)
    jacu_ext_oracle = ExaPF.Jacobian(model_ext_oracle, pf_recourse, mapu)
    constraints_expr = Any[
            ExaPF.ReactivePowerBounds(model_ext_oracle),
        ]
    if line_constraints
            push!(constraints_expr, ExaPF.LineFlows(model_ext_oracle))
    end
    constraints = ExaPF.MultiExpressions(constraints_expr)∘ basis
    jacx_ineq_constraints = ExaPF.Jacobian(model_ext_oracle, constraints, mapx)
    jacu_ineq_constraints = ExaPF.Jacobian(model_ext_oracle, constraints, mapu)
    return (model_ext_oracle,stack_ext_oracle,constraints,jacx_ext_oracle,jacu_ext_oracle,jacx_ineq_constraints,jacu_ineq_constraints) 
end

function Oracle(input_bar,law::LoadProbabilityLaw,batch_size::Integer,aux_loadflow,N_tries)
    """Input bar est le vecteur d'initalisation de la méthode de Newton """
    model_loadflow,stack_loadflow,constraints,jacx_loadflow,_,_,_ = aux_loadflow
    bmin, bmax = ExaPF.bounds(model_loadflow, stack_loadflow);
    gmin,gmax = ExaPF.bounds(model_loadflow, constraints);
    map_x = ExaPF.mapping(model_loadflow, State())
    map_x_no_delta = map_x[1:length(map_x)-1] 
    solver = ExaPF.NewtonRaphson(; verbose=0)
    nbus = length(stack_loadflow.pload);
    success,tries = 0,0;
    pl_batch,ql_batch = [],[];
    N = length(stack_loadflow.input)
    while success<batch_size && (tries<N_tries)
        tries+=1;
        pl,ql = sample(law);
        stack_loadflow.input[1:N] = input_bar;
        stack_loadflow.pload[1:nbus] = pl ;
        stack_loadflow.qload[1:nbus] = ql ;
        ExaPF.set_params!(jacx_loadflow, stack_loadflow);
        ExaPF.nlsolve!(solver, jacx_loadflow, stack_loadflow);
        cons_eval = constraints(stack_loadflow);
        v_bmin,vbmax = maximum((bmin - stack_loadflow.input)[map_x_no_delta]), maximum((stack_loadflow.input - bmax)[map_x_no_delta]);
        v_gmin, v_gmax = maximum(gmin - cons_eval), maximum(cons_eval - gmax);
        error = maximum([v_bmin,vbmax,v_gmin,v_gmax]);
        #println(error)
        if error>tol_ineq
            success+=1
            if length(pl_batch)==0
                pl_batch = pl
                ql_batch = ql
            else
                pl_batch = hcat(pl_batch, pl);
                ql_batch = hcat(ql_batch, ql);
            end
        end
    end
    println("Oracle : tries = ",tries, ", new scenarios = ", success,"." )
    ratio = success/tries
    return pl_batch,ql_batch,ratio
end

function Oracle2(input_bar,law::LoadProbabilityLaw,batch_size::Integer,aux_loadflow,N_tries,threshold)
    """Input bar est le vecteur d'initalisation de la méthode de Newton """
    model_loadflow,stack_loadflow,constraints,jacx_loadflow,_,_,_ = aux_loadflow
    bmin, bmax = ExaPF.bounds(model_loadflow, stack_loadflow);
    gmin,gmax = ExaPF.bounds(model_loadflow, constraints);
    map_x = ExaPF.mapping(model_loadflow, State())
    map_x_no_delta = map_x[1:length(map_x)-1] 
    solver = ExaPF.NewtonRaphson(; verbose=0)
    nbus = length(stack_loadflow.pload);
    success,tries = 0,0;
    pl_batch,ql_batch = [],[];
    N = length(stack_loadflow.input)
    while success<batch_size && (tries<N_tries)
        tries+=1;
        pl,ql = sample(law);
        stack_loadflow.input[1:N] = input_bar;
        stack_loadflow.pload[1:nbus] = pl ;
        stack_loadflow.qload[1:nbus] = ql ;
        ExaPF.set_params!(jacx_loadflow, stack_loadflow);
        ExaPF.nlsolve!(solver, jacx_loadflow, stack_loadflow);
        cons_eval = constraints(stack_loadflow);
        v_bmin,vbmax = maximum((bmin - stack_loadflow.input)[map_x_no_delta]), maximum((stack_loadflow.input - bmax)[map_x_no_delta]);
        v_gmin, v_gmax = maximum(gmin - cons_eval), maximum(cons_eval - gmax);
        error = maximum([v_bmin,vbmax,v_gmin,v_gmax]);
        #println(error)
        if error>threshold
            success+=1
            if length(pl_batch)==0
                pl_batch = pl
                ql_batch = ql
            else
                pl_batch = hcat(pl_batch, pl);
                ql_batch = hcat(ql_batch, ql);
            end
        end
    end
    println("Oracle : tries = ",tries, ", new scenarios = ", success,"." )
    ratio = success/tries
    return pl_batch,ql_batch,ratio,success
end

function Oracle3(input_bar,law::LoadProbabilityLaw,batch_size::Integer,aux_loadflow,N_tries,threshold,threadid)
    """Input bar est le vecteur d'initalisation de la méthode de Newton """
    model_loadflow,stack_loadflow,constraints,jacx_loadflow,_,_,_ = aux_loadflow
    bmin, bmax = ExaPF.bounds(model_loadflow, stack_loadflow);
    gmin,gmax = ExaPF.bounds(model_loadflow, constraints);
    map_x = ExaPF.mapping(model_loadflow, State())
    map_x_no_delta = map_x[1:length(map_x)-1] 
    solver = ExaPF.NewtonRaphson(; verbose=0)
    nbus = length(stack_loadflow.pload);
    success_threshold,success,tries = 0,0,0;
    pl_batch,ql_batch, error_batch = [],[],[];
    pl_log,ql_log,error_log = [],[],[];
    eigvals = [];
    N = length(stack_loadflow.input)
    while success_threshold<batch_size && (tries<N_tries)
        #println(threadid," ",tries," ",success)
        tries+=1;
        pl,ql = sample(law);
        stack_loadflow.input[1:N] = input_bar;
        stack_loadflow.pload[1:nbus] = pl ;
        stack_loadflow.qload[1:nbus] = ql ;
        ExaPF.set_params!(jacx_loadflow, stack_loadflow);
        ExaPF.nlsolve!(solver, jacx_loadflow, stack_loadflow);

        if TEST_EIG_VAL
            jacobian = ExaPF.jacobian!(jacx_loadflow,stack_loadflow);
            JtopJ = jacobian'*jacobian;
            eigvals = [eigvals;(spectral_evaluation(JtopJ))];
        end

        cons_eval = constraints(stack_loadflow);
        v_bmin,vbmax = maximum((bmin - stack_loadflow.input)[map_x_no_delta]), maximum((stack_loadflow.input - bmax)[map_x_no_delta]);
        v_gmin, v_gmax = maximum(gmin - cons_eval), maximum(cons_eval - gmax);
        error = maximum([v_bmin,vbmax,v_gmin,v_gmax]);
        if error>tol_ineq
            success+=1
        end
        if error>threshold
            success_threshold+=1
            if length(pl_batch)==0
                pl_batch = pl
                ql_batch = ql
                error_batch = [error];
            else
                pl_batch = hcat(pl_batch, pl);
                ql_batch = hcat(ql_batch, ql);
                error_batch = [error_batch;error];
            end
        elseif error>tol_ineq
            if length(pl_log)==0
                pl_log = pl;
                ql_log = ql;
                error_log = [error];
            else
                error_log = [error_log;error];
                pl_log = hcat(pl_log, pl);
                ql_log = hcat(ql_log, ql);
            end
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

function OracleParallel(input_bar,law_array,batch_size::Integer,aux_loadflow_array,N_tries,threshold)
    res = [Dict() for i in 1:Kthreads]
    small_batch_size = Integer(max(1,round(batch_size/Kthreads)))
    #for i = 1:Kthreads
    Threads.@threads for i = 1:Kthreads
        res[i] = Oracle3(input_bar,law_array[i],small_batch_size,aux_loadflow_array[i],N_tries/Kthreads,threshold,Threads.threadid())
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