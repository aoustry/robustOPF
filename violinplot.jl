using CSV
using DataFrame
using StatsPlots

df  = DataFrame(CSV.File("resultats_latex_chap6.csv",header=false))
@df df violin(string.(:Column4), log10.(1e-3.+:Column10), linewidth=1,xlabel="Uncertainty standard deviation (s̄)",ylabel = "Cost of robustness (log scale)",yticks=([-3,-2,-1,0,1],["0.001%","0.01%","0.1% ","1% ","10% "]),xticks=([0.5,1.5,2.5],["0.5%","1%","2% "]),legend = false)
png("violinplot.png")
#
plot()

@df df histogram(:Column13,xlabel = "Number of iterations (k)",ylabel = "Number of instances",legend=false,xticks = ([0.5+i for i in 1:9],[string(i) for i in 1:9]))
png("histoIter.png")
