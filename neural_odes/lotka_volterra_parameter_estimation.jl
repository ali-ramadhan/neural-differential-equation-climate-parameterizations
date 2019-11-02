using DifferentialEquations, Plots

function lotka_volterra(dudt, u, p, t)
  x, y = u
  α, β, δ, γ = p

  dudt[1] = dx =  α*x - β*x*y
  dudt[2] = dy = -δ*y + γ*x*y
end

u₀ = [1.0, 1.0]
tspan = (0.0, 10.0)
parameters = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra, u₀, tspan, parameters)
sol = solve(prob, Tsit5())

plot(sol)

using Flux, DiffEqFlux

p = param([2.2, 1.0, 2.0, 0.4]) # Initial Parameter Vector
params = Flux.Params([p])

function predict_adjoint() # Our 1-layer neural network
  diffeq_adjoint(p, prob, Tsit5(), saveat=0.0:0.1:10.0)
end

loss_adjoint() = sum(abs2, x-1 for x in predict_adjoint())

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_adjoint())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.0:0.1:10.0),ylim=(0,6)))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_adjoint, params, data, opt, cb = cb)
