# AutoStep

A collection of implementations of the AutoStep family of samplers.

## Usage

```julia
using Pigeons
using AutoStep

# using AutoStep RWMH
pigeons(
    target = toy_mvn_target(100),
    explorer = SimpleRWMH()
)

# using AutoStep HMC
pigeons(
    target = toy_mvn_target(100),
    explorer = SimpleAHMC()
)

# using AutoStep MALA
pigeons(
    target = toy_mvn_target(100),
    explorer = SimpleAHMC(int_times = AutoStep.FixedIntegrationTime())
)
```