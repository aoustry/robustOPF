using JLD
using HDF5
using Plots

function plotpsi()
    t, val = [],[]
    N = 1e5
    for i in 0:N
        x = -5.0+10*i/N
        f = if x>0 1-exp(-x) else x end
        append!(t,x)
        append!(val,f)
    end
    plot(t,val,label = "Ψ", xlabel = "t",ylabel="Ψ(t)")
    png("plots/psi.png");
end

function plot_time(name,dico)
    p = length(dico["time_master"])
    println("Time master = ", sum(dico["time_master"]));
    println("Time oracle = ", sum(dico["time_oracle"]));
    println("Time (total) = ", sum(dico["time_master"])+sum(dico["time_oracle"]));
    plot()
    plot!(1:p,dico["time_master"],label = "Master pb", xlabel = "Iteration",ylabel="Time (s)")
    plot!(1:p,dico["time_oracle"],label = "Oracle", xlabel = "Iteration",ylabel="Time (s)")
    plot!(1:p,dico["time_master"].+dico["time_oracle"],label = "Total", xlabel = "Iteration",ylabel="Time (s)")
    title!(name);
    png("plots/arobust/"*name*"_time.png");
end

function plot_val(name,dico)
    p = length(dico["time_master"])
    plot()
    plot!(1:p,dico["value_master"], xlabel = "Iteration",ylabel="Value master")
    title!(name);
    png("plots/arobust/"*name*"_val.png");
end

function plot_oracle(name,dico)
    p = length(dico["time_master"])
    plot()
    plot!(1:p,0.00001 .+ dico["stat_oracle"][1:p], xlabel = "Iteration",ylabel="Approximate feas. proba", yaxis= :log)
    title!(name);
    png("plots/arobust/"*name*"_proba.png");
end

name = "pglib_opf_case73_ieee_rts.m_0.05_1.0e-5"
dictionnary = load("logs/"*name*"_logs.jld")["data"]
plot_time(name,dictionnary)
plot_val(name,dictionnary)
plot_oracle(name,dictionnary)