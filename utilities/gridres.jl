using JLD2, FileIO
using Statistics, Core
using Base.Iterators: repeated

function convert_to_low_res(convert_res, data)
    if length(data) == 1
        return data
    else
        @assert length(data) % convert_res == 0
        s = length(data)/convert_res
        temp_array = zeros(convert_res)
        for i in 1:convert_res
            t = data[Int((i-1)*s+1):Int(i*s)]
            temp_array[i] = Statistics.mean(t)
        end
        return temp_array
    end
end

function change_res(convert_res, f)
    low_res_data = Dict()
    for k in keys(f["timeseries"])
        low_res_data[k] = Dict()
        for k_iter in keys(f["timeseries"][k])
            #println(k,k_iter)
            try
                low_res_data[k][k_iter] = convert_to_low_res(convert_res, f["timeseries"][k][k_iter])
            catch y
                println(k,k_iter)
                println(y)
                break
            end
        end
    end
    return low_res_data
end
