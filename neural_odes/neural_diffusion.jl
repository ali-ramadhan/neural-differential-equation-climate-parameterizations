using LinearAlgebra
using Flux, DiffEqFlux, DifferentialEquations, Plots

const N = 16
const L = 1
const Δx = L / N
const κ = 1

 d = -2 * ones(N)
sd = ones(N-1)
A = Array(Tridiagonal(sd, d, sd))
A[1, N] = 1
A[N, 1] = 1
A_diffusion = (κ/Δx^2) .* A

function diffusion(∂u∂t, u, p, t)
    ∂u∂t .= A_diffusion * u
    return 
end

x = range(-L/2, L/2, length=N)
u₀ = @. exp(-100*x^2)
tspan = (0.0, 0.1)
params = [Δx, κ]

datasize = 30
t = range(tspan[1], tspan[2], length=datasize)

prob = ODEProblem(diffusion, u₀, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

dudt = Chain(Dense(N, 100, tanh),
             Dense(100, N))

ps = Flux.params(dudt)
n_ode = x -> neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)

pred = n_ode(u₀)

function predict_n_ode()
  n_ode(u₀)
end

loss_n_ode() = sum(abs2, ode_data .- predict_n_ode())

data = Iterators.repeated((), 1000)
opt = ADAM(0.1)

cb = function ()  # callback function to observe training
  loss = loss_n_ode()
  display(loss)

  # plot current prediction against data
  # cur_pred = Flux.data(predict_n_ode())

  loss < 1 && Flux.stop()
end

cb()
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

