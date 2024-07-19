# autoRWMH

A collection of implementations of the autoRWMH family of samplers.

## Usage

```julia
using Pigeons
using autoRWMH
pigeons(
    target = toy_mvn_target(100),
    explorer = SimpleRWMH()
)
```

## Notes for Ivy: TODO
1. KS test
2. simulations using InferHub, InferHub-infra (with nextflow)
3. Intro: RWMH is useful because of non-differentiable potentials()
4. bibtex + comments