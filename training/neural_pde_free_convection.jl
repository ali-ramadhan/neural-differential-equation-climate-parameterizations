using Printf
using Random
using Statistics
using LinearAlgebra

using Flux
using DifferentialEquations
using DiffEqFlux

using JLD2
using BSON
using Plots

using Flux: @epochs

#####
##### Load data from JLD2 file
#####

file = jldopen("../data/ocean_convection_profiles.jld2")

Is = keys(file["timeseries/t"])

Nz = file["grid/Nz"]
Lz = file["grid/Lz"]
Nt = length(Is)

t = zeros(Nt)
T = T_data = zeros(Nt, Nz)
wT = zeros(Nt, Nz)

for (i, I) in enumerate(Is)
    t[i] = file["timeseries/t/$I"]
    T[i, :] = file["timeseries/T/$I"][1, 1, 2:Nz+1]
    wT[i, :] = file["timeseries/wT/$I"][1, 1, 2:Nz+1]
end

#####
##### Plot animation of T(z,t) from data
#####

z = file["grid/zC"]
#=
anim = @animate for n=1:10:Nt
    t_str = @sprintf("%.2f", t[n] / 86400)
    plot(T[n, :], z, linewidth=2,
         xlim=(19, 20), ylim=(-100, 0), label="",
         xlabel="Temperature (C)", ylabel="Depth (z)",
         title="Deepening mixed layer: $t_str days", show=false)
end

gif(anim, "deepening_mixed_layer.gif", fps=15)
=#
#####
##### Coarse grain data to 32 vertical levels (plus halo regions)
#####

function coarse_grain(data, resolution)
    @assert length(data) % resolution == 0
    s = length(data) / resolution
    
    data_cs = zeros(resolution)
    for i in 1:resolution
        t = data[Int((i-1)*s+1):Int(i*s)]
        data_cs[i] = mean(t)
    end
    
    return data_cs
end

coarse_resolution = cr = 32

T_cs = zeros(coarse_resolution+2, Nt)
wT_cs = zeros(coarse_resolution+2, Nt)

z_cs = coarse_grain(collect(z), cr)

for n=1:Nt
    T_cs[2:end-1, n] .= coarse_grain(T[n, :], coarse_resolution)
    wT_cs[2:end-1, n] .= coarse_grain(wT[n, :], coarse_resolution)
end

# Fill halo regions to enforce boundary conditions.
T_cs[1,   :] .= T_cs[2,     :]
T_cs[end, :] .= T_cs[end-1, :]

wT_cs[1,   :] .= wT_cs[2,     :]
wT_cs[end, :] .= wT_cs[end-1, :]

#####
##### Plot coarse temperature and wT profiles
#####
#=
@info "Plotting coarse temperature profile..."

anim = @animate for n=1:10:Nt
    t_str = @sprintf("%.2f", t[n] / 86400)
    plot(T_cs[2:cr+1, n], z_cs, linewidth=2,
         xlim=(19, 20), ylim=(-100, 0), label="",
         xlabel="Temperature (C)", ylabel="Depth (z)",
         title="Deepening mixed layer: $t_str days", show=false)
end

gif(anim, "deepening_mixed_layer_T_coarse.gif", fps=15)

@info "Plotting coarse wT profile..."

anim = @animate for n=1:10:Nt
    t_str = @sprintf("%.2f", t[n] / 86400)
    plot(wT_cs[2:cr+1, n], z_cs, linewidth=2,
         xlim=(-1e-4, 1e-4), ylim=(-100, 0), label="",
         xlabel="Temperature (C)", ylabel="Depth (z)",
         title="Deepening mixed layer: $t_str days", show=false)
end

gif(anim, "deepening_mixed_layer_wT_coarse.gif", fps=15)
=#
#####
##### Generate differentiation matrices
#####

cr_Δz = Lz / cr  # Coarse resolution Δz

# Dzᶠ computes the derivative from cell center to cell (F)aces
Dzᶠ = 1/cr_Δz * Tridiagonal(-ones(cr+1), ones(cr+2), zeros(cr+1))

# Dzᶜ computes the derivative from cell faces to cell (C)enters
Dzᶜ = 1/cr_Δz * Tridiagonal(zeros(cr+1), -ones(cr+2), ones(cr+1))

# Impose boundary condition that derivative goes to zero at top and bottom.
Dzᶠ[1, 1] = 0
Dzᶜ[cr, cr] = 0

#####
##### Create training data
#####

Tₙ   = zeros(cr+2, Nt-1)
Tₙ₊₁ = zeros(cr+2, Nt-1)
wTₙ  = zeros(cr+2, Nt-1)
∂zTₙ = zeros(cr+2, Nt-1)

for i in 1:Nt-1
       Tₙ[:, i] .=  T_cs[:,   i]
     Tₙ₊₁[:, i] .=  T_cs[:, i+1]
      wTₙ[:, i] .= wT_cs[:,   i]
     ∂zTₙ[:, i] .= Dzᶠ * T_cs[:, i]
end

N_skip = 0  # Skip first N_skip iterations to avoid learning transients?
N = 32  # Number of training data pairs.

rinds = randperm(Nt-N_skip)[1:N]

pre_training_data = [(∂zTₙ[:, i], wTₙ[:, i]) for i in 1:N]
training_data = [(Tₙ[:, i], Tₙ₊₁[:, i]) for i in 1:N]

#####
##### Create neural network
#####

# Complete black box right-hand-side.
#  dTdt_NN = Chain(Dense(cr+2,  2cr, tanh),
#                  Dense(2cr,  cr+2))

# Use NN to parameterize a diffusivity or κ profile.
dTdt_NN = Chain(T -> Dzᶠ*T,
               Dense(cr+2,  2cr, tanh),
               Dense(2cr, 2cr, tanh),
               Dense(2cr,  cr+2),
               NNDzT -> Dzᶜ * NNDzT)

NN_params = Flux.params(dTdt_NN)

#####
##### Pre-train the neural network on (T, wT) data pairs
#####

pre_loss_function(∂zTₙ, wTₙ) = sum(abs2, dTdt_NN(∂zTₙ) .- wTₙ)

popt = ADAM(0.01)

function precb()
    loss = sum(abs2, [pre_loss_function(pre_training_data[i]...) for i in 1:N-1])
    println("loss = $loss")
end

pre_train_epochs = 5
for _ in 1:pre_train_epochs
    Flux.train!(pre_loss_function, NN_params, pre_training_data, popt, cb = Flux.throttle(precb, 5))
end

#####
##### Define loss function
#####

tspan = (0.0, 600.0)  # 10 minutes
neural_pde_prediction(T₀) = neural_ode(dTdt_NN, T₀, tspan, Tsit5(), reltol=1e-4, save_start=false, saveat=tspan[2])

loss_function(Tₙ, Tₙ₊₁) = sum(abs2, Tₙ₊₁ .- neural_pde_prediction(Tₙ))

#####
##### Choose optimization algorithm
#####

opt = ADAM(1e-3)

#####
##### Callback function to observe training.
#####

function cb()
    train_loss = sum([loss_function(Tₙ[:, i], Tₙ₊₁[:, i]) for i in 1:N])

    nn_pred = neural_ode(dTdt_NN, Tₙ[:, 1], (t[1], t[N]), Tsit5(), saveat=t[1:N], reltol=1e-4) |> Flux.data
    test_loss = sum(abs2, T_cs[:, 1:N] .- nn_pred)
    
    println("train_loss = $train_loss, test_loss = $test_loss")
    return train_loss
end

#####
##### Train!
#####

epochs = 10
best_loss = Inf
last_improvement = 0

for epoch_idx in 1:epochs
    global best_loss, last_improvement

    @info "Epoch $epoch_idx"
    Flux.train!(loss_function, NN_params, training_data, opt, cb=cb) # cb=Flux.throttle(cb, 10))
    
    loss = cb()

    if loss <= best_loss
        @info("Record low loss! Saving neural network out to dTdt_NN.bson")
        BSON.@save "dTdt_NN.bson" dTdt_NN
        best_loss = loss
        last_improvement = epoch_idx
    end
   
    # If we haven't seen improvement in 2 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 2 && opt.eta > 1e-6
        opt.eta /= 5.0
        @warn("Haven't improved in a while, dropping learning rate to $(opt.eta)")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end 
end

#####
##### Run the neural PDE forward to see how well it performs just by itself.
#####

nn_pred = neural_ode(dTdt_NN, Tₙ[:, 1], (t[1], t[end]), Tsit5(), saveat=t, reltol=1e-4) |> Flux.data

z_cs = coarse_grain(z, cr)

anim = @animate for n=1:10:Nt
    t_str = @sprintf("%.2f", t[n] / 86400)
    plot(T_cs[2:end-1, n], z_cs, linewidth=2,
         xlim=(19, 20), ylim=(-100, 0), label="Data",
         xlabel="Temperature (C)", ylabel="Depth (z)",
         title="Deepening mixed layer: $t_str days",
         legend=:bottomright, show=false)
    plot!(nn_pred[2:end-1, n], z_cs, linewidth=2, label="Neural PDE", show=false)
end

gif(anim, "deepening_mixed_layer_neural_PDE.gif", fps=15)

