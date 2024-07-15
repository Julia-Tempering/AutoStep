using Pigeons
using BridgeStan
using autoRWMH
pigeons(
    target = toy_mvn_target(5),
    explorer = SimpleRWMH()
)