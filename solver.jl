function log_info(logs,tmaster,toracle,vmaster,statoracle,sizescen)
    logs["time_master"] = [logs["time_master"]; tmaster];
    logs["time_oracle"] = [logs["time_oracle"]; toracle];
    logs["value_master"] = [logs["value_master"]; vmaster];
    logs["stat_oracle"] = [logs["stat_oracle"]; statoracle];
    logs["size_scenarios"] = [logs["size_scenarios"]; sizescen];
end
time_from_now(seconds) = round(Int, 10^9 * seconds + time_ns())

function load_uncertainty_matrix_and_scale_factor(name,nbus)
    if name*".jld" in readdir("data/")
        A = load("data/"*name*".jld")["data"] ;
        println("Loading matrix A")  ;
        magnitude = (load("data/"*name*"_mag.jld")["data"]);
        println(magnitude)
        return A, magnitude
    end
        println("Creating matrix A")
        Random.seed!(0000);
        M= rand(nbus,nbus);
        Q, _ = qr(M);
        A = copy(Q[:,1:min(nbus,8)]);
        save("data/"*name*".jld", "data", A);
        target = Float(readline())
        calibration(case,target)
        return load_uncertainty_matrix_and_scale_factor(name,nbus)
end

function iterative_discretization(case,line_constraints,scaling)
    Random.seed!(0000);
    """Problem Data """
    model = ExaPF.PolarFormRecourse(case,1);
    basis = ExaPF.PolarBasis(model);
    map_x = ExaPF.mapping(model, State())
    map_x_no_delta = map_x[1:length(map_x)-1]
    map_u = ExaPF.mapping(model, Control())

    mapmain = [map_x_no_delta; map_u];
    nx,nu = length(map_x), length(map_u)
    N = nx -1 + nu;
    ngen = PS.get(model, PS.NumberOfGenerators())
    nbus = PS.get(model, PS.NumberOfBuses())
    stack_refcase = ExaPF.NetworkStack(model)
    pload_refcase = stack_refcase.pload |> Array
    qload_refcase = stack_refcase.qload |> Array
    evaluators = [OracleEvaluator(model,1; line_constraints=line_constraints) for i in 1:Kthreads]
    array = split(case,"/")

    if GaussianLowRand
        A,scale_factor = load_uncertainty_matrix_and_scale_factor(array[length(array)],nbus)
        magnitude = percentage_std * scale_factor
        law_array = [GaussianIndependantLowRank(A,size(A,2),pload_refcase,qload_refcase,magnitude) for i in 1:Kthreads]
        avgstd = compute_std_sample(law_array[1],100000)
        avgstd = round(avgstd*1000)/1000;
        println("Law std = ",avgstd)

    else
        println("Standard error remains to be computed")
        magnitude  = avgstd  = percentage_std*0.01
        C = (1/nbus)* ones((nbus,nbus))
        for i in 1:nbus
            C[i,i] = 1.0
        end
        dis_array = [MvNormal(zeros(nbus), C) for i in 1:Kthreads]
        law_array = [GaussianEqualCorrelation(dis_array[i],C,pload_refcase,qload_refcase,magnitude) for i in 1:Kthreads]
    end

    logs = Dict("objective_scaling"=>scaling,"line_constraints"=>line_constraints,"avgstd"=>avgstd,"tol_ineq" => tol_ineq, "Ntries" => Ntries, "K" =>  Kthreads, "magnitude" => magnitude,"time_master"=>[], "time_oracle"=>[],"value_master"=> [],"stat_oracle"=> [],"size_scenarios"=> []);

    """Data structures """
    stack_main = ExaPF.NetworkStack(model)
    """ Initialization with random scenarios, solve the batch"""
    p_total,q_total = hcat(pload_refcase), hcat(qload_refcase);
    warm_start_dict = Dict();
    warm_start_dict["bool"] = false;
    mtime = @elapsed begin
        output = (solve_corrective_opf_ipopt(case,p_total,q_total,line_constraints,warm_start_dict,barrier_min,scaling))
        # output = (solve_corrective_opf(case,p_total,q_total,line_constraints,warm_start_dict,barrier_min))
        obj_value_no_sip = output.obj_val;
    end
    log_info(logs,mtime,0.0,obj_value_no_sip,1,size(p_total,2));
    sol_master = output.model
    sol_master_stack = sol_master.stack
    logs["value_no_robust"] = obj_value_no_sip
    eigvals = []
    stack_main.input[1:nbus] .= sol_master_stack.vmag[1:nbus]
    stack_main.input[nbus+1:2*nbus] .= sol_master_stack.vang[1:nbus]
    stack_main.input[2*nbus+2+ngen:end] .= sol_master_stack.pgen[1:ngen]
    outer_iter_counter = 0
    counter_infeas = 0
    threshold = 1e3 * tol_ineq
    obj_value = obj_value_no_sip
    n_newbatch = 0;
    end_time = time_from_now(5*3600)
    while outer_iter_counter<MAXIT
        outer_iter_counter +=1
        otime = @elapsed begin
            println("Threshold = ",threshold);
            res = OracleParallel(model, evaluators, stack_main.input,law_array,size_batch,Ntries,threshold)
        end
        p_newbatch = res["pl_batch"];
        q_newbatch = res["ql_batch"];
        measured_ratio = res["ratio"];
        eigvals = [eigvals; res["eigvals"]]
        println("empirical frequency = ",measured_ratio)
        n_newbatch = res["len"];
        boolean = res["boolean"];
        if !(boolean)
            threshold = max(tol_ineq,threshold/4.0);
        end
        if res["success"] == 0
            log_info(logs,0.0,otime,obj_value,measured_ratio,size(p_total,2));
            array = split(case,"/");
            name = array[length(array)]
            logs["status"] = "success"
            save("logs/"*name*"_"*string(avgstd)*"_"*string(p0)*"_"*string(GaussianLowRand)*"_logs.jld", "data", logs)
            return
        end
        if time_ns() >= end_time
            array = split(case,"/");
            name = array[length(array)]
            logs["status"] = "TimeOut"
            save("logs/"*name*"_"*string(avgstd)*"_"*string(p0)*"_"*string(GaussianLowRand)*"_logs.jld", "data", logs)
            return
        end
        p_total,q_total = hcat(p_total, p_newbatch), hcat(q_total, q_newbatch);
        #Build initial x0
        warm_start_dict = Dict();
        warm_start_dict["bool"] = WARMSTART;
        if WARMSTART
            x_batches = EvalXbatch(model, evaluators[1], stack_main.input,n_newbatch,p_newbatch,q_newbatch);
            warm_start_dict["x0"] = vcat(sol_master_stack.input[sol_master.blk_mapx],x_batches,sol_master_stack.input[sol_master.blk_mapu[1:nu]])
        end
        mtime = @elapsed begin
            # output = (solve_corrective_opf(case,p_total,q_total,line_constraints,warm_start_dict,barrier_min))
            output = (solve_corrective_opf_ipopt(case,p_total,q_total,line_constraints,warm_start_dict,barrier_min,scaling))
        end
        if output.status==MOI.LOCALLY_INFEASIBLE
            counter_infeas+=1;
            if counter_infeas>=5
                array = split(case,"/");
                name = array[length(array)]
                logs["status"] = "Infeas"
                save("logs/"*name*"_"*string(avgstd)*"_"*string(p0)*"_"*string(GaussianLowRand)*"_logs.jld", "data", logs)
                return
            end
        end
        sol_master = output.model
        sol_master_stack = output.model.stack
        stack_main.input[1:nbus] .= sol_master_stack.vmag[1:nbus]
        stack_main.input[nbus+1:2*nbus] .= sol_master_stack.vang[1:nbus]
        stack_main.input[2*nbus+2+ngen:end] .= sol_master_stack.pgen[1:ngen]
        @info(stack_main.pgen[1:10])
       ##########################################################################################################################
        obj_value = output.obj_val;
        println("Obj value = ", obj_value)
        log_info(logs,mtime,otime,output.obj_val,measured_ratio,size(p_total,2));
        #print((eigvals))
        if TEST_EIG_VAL
            println("Min/max eigval ",minimum(eigvals),"/",maximum(eigvals))
        end
    end
    logs["status"] = "MaxIt"
    array = split(case,"/");
    name = array[length(array)]
    save("logs/"*name*"_"*string(avgstd)*"_"*string(p0)*"_"*string(GaussianLowRand)*"_logs.jld", "data", logs)
end
