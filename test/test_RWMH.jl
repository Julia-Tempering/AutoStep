using Pigeons
using BridgeStan
using autoRWMH

pigeons(
    target = toy_mvn_target(5),
    explorer = AutoMALA(),
    record = [traces; record_default()]
)
pigeons(
    target = toy_mvn_target(1),
    explorer = SimpleRWMH(),
    record = [traces; record_default()]
)