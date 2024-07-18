using Pigeons
using BridgeStan
using autoRWMH

pigeons(
    target = toy_mvn_target(1),
    explorer = SimpleRWMH(),
    record = record_default()
)