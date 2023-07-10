using JLD
using ExaPF
using LinearAlgebra
using Distributions
using MathOptInterface 

include("corrective_opf.jl")
include("evaluation.jl")
include("probalaw.jl")
include("oracle.jl")


""" Problem Parameters """
const tol_ineq = 1e-4;
const p0 = 1e-4;
const αlevel = 1e-2;
const Ntries = round(log(αlevel)/log(1-p0));
const barrier_min = 1e-4;
const size_batch = 5;
const Kthreads = 12;
const MAXIT = 25;
const SMOOTHING_PARAM = 1e-3;
const WARMSTART = true ;
TEST_EIG_VAL = false
if Kthreads!=1
    TEST_EIG_VAL = false
end
GaussianLowRand = true
percentage_std = 0.5
linelim = false
#@assert(Threads.nthreads()>=Kthreads);

time_from_now(seconds) = round(Int, 10^9 * seconds + time_ns())

function log_info(logs,tmaster,toracle,vmaster,statoracle,sizescen)
    logs["time_master"] = [logs["time_master"]; tmaster];
    logs["time_oracle"] = [logs["time_oracle"]; toracle];
    logs["value_master"] = [logs["value_master"]; vmaster];
    logs["stat_oracle"] = [logs["stat_oracle"]; statoracle];
    logs["size_scenarios"] = [logs["size_scenarios"]; sizescen];
end

function load_uncertainty_matrix_and_scale_factor(name,nbus)
    if name*".jld" in readdir("data/")
        A = load("data/"*name*".jld")["data"] ;  
        println("Loading matrix A")  ; 
        magnitude = (load("data/"*name*"_mag.jld")["data"]);
        return A,magnitude
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
    map_x = ExaPF.mapping(model, State()); map_x_no_delta = map_x[1:length(map_x)-1] ; map_u = ExaPF.mapping(model, Control());
    mapmain = [map_x_no_delta;map_u];
    nx,nu = length(map_x), length(map_u); N = nx -1 + nu;
    ngen = PS.get(model, PS.NumberOfGenerators())
    nbus = PS.get(model, PS.NumberOfBuses())
    stack_refcase = ExaPF.NetworkStack(model)
    pload_refcase = stack_refcase.pload |> Array
    qload_refcase = stack_refcase.qload |> Array
    aux_loadflow_array = [build_aux_loadflow(model,basis,line_constraints) for i in 1:Kthreads]
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
    println(output)
    obj_value_no_sip = output.obj_val;
    end
    log_info(logs,mtime,0.0,obj_value_no_sip,1,size(p_total,2));
    sol_master = output.model
    sol_master_stack = sol_master.stack
    logs["value_no_robust"] = obj_value_no_sip
    eigvals = []
    stack_main.input[1:nbus] .= sol_master_stack.vmag[1:nbus]
    stack_main.input[nbus+1:2*nbus] .= sol_master_stack.vang[1:nbus]
    stack_main.input[2*nbus+2+ngen:end] .=sol_master_stack.pgen[1:ngen]
    outer_iter_counter = 0 ;
    counter_infeas = 0; 
    threshold = 1e3 * tol_ineq;
    obj_value = obj_value_no_sip;
    n_newbatch = 0;
    end_time = time_from_now(5*3600)
    while outer_iter_counter<MAXIT
        outer_iter_counter +=1
        otime = @elapsed begin
        println("Threshold = ",threshold);
        res = OracleParallel(stack_main.input,law_array,size_batch,aux_loadflow_array,Ntries,threshold)
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
            x_batches = EvalXbatch(stack_main.input,n_newbatch,p_newbatch,q_newbatch,aux_loadflow_array[1]);
            warm_start_dict["x0"] = vcat(sol_master_stack.input[sol_master.blk_mapx],x_batches,sol_master_stack.input[sol_master.blk_mapu[1:nu]])
        end
        mtime = @elapsed begin
        #output = (solve_corrective_opf(case,p_total,q_total,line_constraints,warm_start_dict,barrier_min))
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
        stack_main.input[2*nbus+2+ngen:end] .=sol_master_stack.pgen[1:ngen]
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

#case = "../matpower/data/case9.m"
#case = "../matpower/data/case14.m"
#case = "../matpower/data/case30.m"
#case = "../matpower/data/case57.m"
#case = "../matpower/data/case118.m"
#case = "../matpower/data/case300.m"
#case = "../matpower/data/case1354pegase.m"
#case = pglibfolder*"pglib_opf_case793_goc.m"

#case =pglibfolder*"pglib_opf_case3_lmbd.m" #oktrue
#case =pglibfolder*"pglib_opf_case5_pjm.m" #oktrue
#case =pglibfolder*"pglib_opf_case14_ieee.m" #oktrue
#case =pglibfolder*"pglib_opf_case24_ieee_rts.m" #oktrue
#case =pglibfolder*"pglib_opf_case30_as.m" #oktrue
#case =pglibfolder*"pglib_opf_case30_ieee.m" #infeasible with true, ok with false
#case =pglibfolder*"pglib_opf_case39_epri.m" #oktrue
#case =pglibfolder*"pglib_opf_case57_ieee.m"  #infeasible with true
#case =pglibfolder*"pglib_opf_case1888_rte.m"
#case =pglibfolder*"pglib_opf_case1951_rte.m"
#case =pglibfolder*"pglib_opf_case2869_pegase.m"

pglibfolder = "../../pglib-opf/"
#Normal conditions
case =pglibfolder*"pglib_opf_case3_lmbd.m"
iterative_discretization(case,linelim,.1)
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case5_pjm.m"
iterative_discretization(case,linelim,.1) 
case =pglibfolder*"pglib_opf_case14_ieee.m"
iterative_discretization(case,linelim,.1)
#=case =pglibfolder*"pglib_opf_case24_ieee_rts.m"
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case30_as.m"
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case30_ieee.m"
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case39_epri.m"
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case57_ieee.m"
iterative_discretization(case,linelim,.01) 
case =pglibfolder*"pglib_opf_case73_ieee_rts.m"
iterative_discretization(case,linelim,.1) 
case =pglibfolder*"pglib_opf_case89_pegase.m"
iterative_discretization(case,linelim,.001)
case =pglibfolder*"pglib_opf_case118_ieee.m"
iterative_discretization(case,linelim,.01)
case =pglibfolder*"pglib_opf_case179_goc.m" 
iterative_discretization(case,linelim,.01)
case =pglibfolder*"pglib_opf_case200_activ.m"
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case240_pserc.m"
iterative_discretization(case,linelim,.01)
case =pglibfolder*"pglib_opf_case300_ieee.m" ; 
iterative_discretization(case,linelim,.001)
case =pglibfolder*"pglib_opf_case588_sdet.m"
iterative_discretization(case,linelim,.05)
case =pglibfolder*"pglib_opf_case1354_pegase.m"
iterative_discretization(case,linelim,.01) 


#API conditions
case =pglibfolder*"api/pglib_opf_case3_lmbd__api.m"
iterative_discretization(case,linelim,.1) #Ok
case =pglibfolder*"api/pglib_opf_case5_pjm__api.m"
iterative_discretization(case,linelim,.1) #Ok
case =pglibfolder*"api/pglib_opf_case14_ieee__api.m"
iterative_discretization(case,linelim,.1) #Ok
case =pglibfolder*"api/pglib_opf_case24_ieee_rts__api.m"
iterative_discretization(case,linelim,.1) #Ok
case =pglibfolder*"api/pglib_opf_case30_as__api.m"
iterative_discretization(case,linelim,.1) #ok
case =pglibfolder*"api/pglib_opf_case30_ieee__api.m"
iterative_discretization(case,linelim,.1) #Ok
case =pglibfolder*"api/pglib_opf_case39_epri__api.m"
iterative_discretization(case,linelim,.01) #Ok
case =pglibfolder*"api/pglib_opf_case57_ieee__api.m"
iterative_discretization(case,linelim,.01) #ok
case =pglibfolder*"api/pglib_opf_case73_ieee_rts__api.m"
iterative_discretization(case,linelim,.01) #ok 
case =pglibfolder*"api/pglib_opf_case89_pegase__api.m"
iterative_discretization(case,linelim,.005) #Ok
case =pglibfolder*"api/pglib_opf_case118_ieee__api.m"
iterative_discretization(case,linelim,.01) #Ok
case =pglibfolder*"api/pglib_opf_case200_activ__api.m"
iterative_discretization(case,linelim,.1) #Ok
case =pglibfolder*"api/pglib_opf_case240_pserc__api.m"
iterative_discretization(case,linelim,.01)#Ok
case =pglibfolder*"api/pglib_opf_case300_ieee__api.m" ; 
iterative_discretization(case,linelim,.001)#Ok
case =pglibfolder*"api/pglib_opf_case588_sdet__api.m"
iterative_discretization(case,linelim,.01)#Ok
case =pglibfolder*"api/pglib_opf_case1354_pegase__api.m"
iterative_discretization(case,linelim,.01) #Ok 

case =pglibfolder*"api/pglib_opf_case179_goc__api.m" 
iterative_discretization(case,linelim,0.01) #Difficult: singularException pour incertitude 0.01, Ok pour incertitude pour 0.005

=#







