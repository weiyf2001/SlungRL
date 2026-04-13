from train.experiments.policies.lstm_history import POLICY_SPEC as LSTM_HISTORY
from train.experiments.policies.mlp_history import POLICY_SPEC as MLP_HISTORY
from train.experiments.policies.mlp_memoryless import POLICY_SPEC as MLP_MEMORYLESS
from train.experiments.policies.physics_guided_belief import POLICY_SPEC as PHYSICS_GUIDED_BELIEF


POLICY_SPECS = {
    LSTM_HISTORY.name: LSTM_HISTORY,
    MLP_HISTORY.name: MLP_HISTORY,
    MLP_MEMORYLESS.name: MLP_MEMORYLESS,
    PHYSICS_GUIDED_BELIEF.name: PHYSICS_GUIDED_BELIEF,
}
