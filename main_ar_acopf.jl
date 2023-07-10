using JLD
using ExaPF
using LinearAlgebra
using Distributions
using MathOptInterface 

include("corrective_opf.jl")
include("evaluation.jl")
include("probalaw.jl")
include("oracle.jl")
include("solver.jl")

""" Data folder """
pglibfolder = "../../pglib-opf/"


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

#Normal conditions
case =pglibfolder*"pglib_opf_case3_lmbd.m"
iterative_discretization(case,linelim,.1)
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case5_pjm.m"
iterative_discretization(case,linelim,.1) 
case =pglibfolder*"pglib_opf_case14_ieee.m"
iterative_discretization(case,linelim,.1)
case =pglibfolder*"pglib_opf_case24_ieee_rts.m"
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
iterative_discretization(case,linelim,0.01) #Difficult: singularException 







