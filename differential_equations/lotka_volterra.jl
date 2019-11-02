using DifferentialEquations, Plots

function lotka_volterra(dudt, u, p, t)
  x, y = u
  α, β, δ, γ = p

  dudt[1] = dx =  α*x - β*x*y
  dudt[2] = dy = -δ*y + γ*x*y
end

u₀ = [1.0, 1.0]
tspan = (0.0, 10.0)
params = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra, u₀, tspan, params)
sol = solve(prob, Tsit5())

plot(sol)
