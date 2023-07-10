using JLD
using HDF5
using Plots
using PrettyTables


function summary_table(proba_string, proba_float)
    #names, std, it_number, final_proba, time_master, time_oracle, time_total, value_progress, final_scen = [],[],[],[],[],[],[],[],[]
    data = ["test" 0 0 0 0 0 0 0  0]

    folder = "logs/"
    name_list = sort(readdir(folder))

    for file in name_list
        dico = load(folder*file)["data"]

        array = split(file,"_")
        array2 = split(file,".m")
        n = parse(Int,String(split(array[3],"e")[2]))
        m = length(dico["time_oracle"])
        target_proba = (String(array[length(array)-2]))
        if target_proba == proba_string
            evol = round(10000*(dico["value_master"][m]-dico["value_no_robust"])/dico["value_no_robust"])/100
            data = [data; array2[1] dico["avgstd"]  dico["status"] evol m-1 dico["size_scenarios"][m] round(sum(dico["time_master"])) round(sum(dico["time_oracle"])) round(sum(dico["time_oracle"])+sum(dico["time_master"])) ]

            #= append!(names,array2[1])
            append!(std,array[length(array)-2])
            append!(it_number,m)
            append!(final_proba,dico["stat_oracle"][m]<proba_float)
            append!(time_master,sum(dico["time_master"]))
            append!(time_oracle,sum(dico["time_oracle"]))
            append!(time_total,sum(dico["time_oracle"])+sum(dico["time_master"]))
            append!(value_progress,100*(dico["value_master"][m]-dico["value_no_robust"])/dico["value_no_robust"])
            append!(final_scen,dico["size_scenarios"][m]) =#
        end
    end
    return pretty_table(data[2:end,:]; header = ["Instance", "Std","Success","Rob. cost","It. nbr.","Scen.", "Master (s)","Oracle (s)","Total (s)"], backend = Val(:latex))
end


function probacurves(proba_string, proba_float)
    #names, std, it_number, final_proba, time_master, time_oracle, time_total, value_progress, final_scen = [],[],[],[],[],[],[],[],[]
    data = ["test" 0 0 0 0 0 0 0  0]

    folder = "logs/log_final/"
    name_list = sort(readdir(folder))
    plot()
    for file in name_list
        dico = load(folder*file)["data"]

        array = split(file,"_")
        array2 = split(file,".m")
        n = parse(Int,String(split(array[3],"e")[2]))
        m = length(dico["time_oracle"])
        target_proba = (String(array[length(array)-2]))
        if target_proba == proba_string
            
            if dico["status"]=="success"
            k = length(dico["stat_oracle"]);
            plot!(1:k,(dico["stat_oracle"].+1e-5),yscale = :log10,legend = false,color = n)
            end
        end
    end
    xlabel!("Iteration index")
    ylabel!("Probability of {G(x,y)>ϵ}")
    png("stat_curve.png")
    return #pretty_table(data[2:end,:]; header = ["Instance", "Std","Success","Rob. cost","It. nbr.","Scen.", "Master (s)","Oracle (s)","Total (s)"], backend = Val(:latex))
end

summary_table("0.0001",0.0001)
