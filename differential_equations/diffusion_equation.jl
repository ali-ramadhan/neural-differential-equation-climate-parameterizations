using DifferentialEquations, Plots

function diffusion(∂u∂t, u, p, t)
    @inbounds begin
        ∂u∂t[1] = κ * (u[N] -2u[1] + u[2]) / Δx
        for i in 2:N-1
            ∂u∂t[i] = κ * (u[i-1] -2u[i] + u[i+1]) / Δx
        end
        ∂u∂t[N] = κ * (u[N-1] -2u[N] + u[1]) / Δx
    end

    return 
end

const N = 128
const L = 1
const Δx = L / N
const κ = 1

x = range(-L/2, L/2, length=N)
u₀ = @. exp(-100*x^2)
tspan = (0.0, 1.0)
params = [Δx, κ]

prob = ODEProblem(diffusion, u₀, tspan, params)
sol = solve(prob, Tsit5())

plot(x, sol.u[1], lab="t = 0")
plot!(x, sol.u[end], lab="t = 1")
title!("1D diffusion equation")

