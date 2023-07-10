using JLD
using ExaPF
using LinearAlgebra
include("corrective_opf.jl")
include("SQPsteps.jl")
include("evaluation.jl")
include("probalaw.jl")
include("oracle.jl")

function load_uncertainty_matrix(name,nbus)
    if name*".jld" in readdir("data/")
        A = load("data/"*name*".jld")["data"] ;  
    else
        Random.seed!(0002);
        M= rand(nbus,nbus);
        Q, _ = qr(M);
        A = copy(Q[:,1:min(nbus,8)]);
        save("data/"*name*".jld", "data", A);
    end
    return A
end

function calibration(case,target)
    model = ExaPF.PolarFormRecourse(case,1);
    nbus = PS.get(model, PS.NumberOfBuses())
    array = split(case,"/")
    stack_refcase = ExaPF.NetworkStack(model)
    pload_refcase = stack_refcase.pload |> Array
    qload_refcase = stack_refcase.qload |> Array
    A = load_uncertainty_matrix(array[length(array)],nbus)
    law = GaussianIndependantLowRank(A,size(A,2),pload_refcase,qload_refcase,1.0) 
    r = compute_std_sample(law,1000000)
    magnitude = target/r
    array = split(case,"/")
    name = array[length(array)]
    save("data/"*name*"_mag.jld", "data", magnitude)
end


#=case ="../pglib-opf/pglib_opf_case3_lmbd.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case5_pjm.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case14_ieee.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case24_ieee_rts.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case30_as.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case30_ieee.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case39_epri.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case57_ieee.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case73_ieee_rts.m"
calibration(case,0.01) 
case ="../pglib-opf/pglib_opf_case89_pegase.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case118_ieee.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case179_goc.m" 
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case200_activ.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case240_pserc.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case300_ieee.m" ; 
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case588_sdet.m"
calibration(case,0.01)
case ="../pglib-opf/pglib_opf_case1354_pegase.m"
calibration(case,0.01) 
case = "../matpower/data/case1888rte.m"
calibration(case,0.01) 
case ="../pglib-opf/pglib_opf_case2000_goc.m"
calibration(case,0.01) 



case ="../pglib-opf/api/pglib_opf_case3_lmbd__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case5_pjm__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case14_ieee__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case24_ieee_rts__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case30_as__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case30_ieee__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case39_epri__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case57_ieee__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case73_ieee_rts__api.m"
calibration(case,0.01) 
case ="../pglib-opf/api/pglib_opf_case89_pegase__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case118_ieee__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case179_goc__api.m" 
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case200_activ__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case240_pserc__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case300_ieee__api.m" ; 
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case588_sdet__api.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case1354_pegase__api.m"
calibration(case,0.01)


=#


case ="../pglib-opf/pglib_opf_case588_sdet.m"
calibration(case,0.01)
case ="../pglib-opf/api/pglib_opf_case588_sdet__api.m"
calibration(case,0.01)