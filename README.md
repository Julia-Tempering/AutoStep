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